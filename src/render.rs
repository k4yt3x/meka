use std::io::{self, Write};

use crossterm::style::{Color, Stylize};
use termimad::MadSkin;

// ---------------------------------------------------------------------------
// Output spacing state machine
// ---------------------------------------------------------------------------

enum LastOutput {
    Nothing,
    Prompt,
    Text,
    Thinking,
    ToolIndicator,
    TodoList,
}

/// Tracks what was last printed and decides whether a blank line is needed
/// before the next output. Replaces ad-hoc spacing flags.
pub struct OutputSpacing {
    last: LastOutput,
}

impl OutputSpacing {
    pub fn new() -> Self {
        Self {
            last: LastOutput::Nothing,
        }
    }

    /// Call before printing streamed text. Returns true if a blank line
    /// should be emitted first.
    pub fn before_text(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::ToolIndicator | LastOutput::Thinking);
        self.last = LastOutput::Text;
        need_blank
    }

    /// Call before printing a tool indicator. Returns true if a blank line
    /// should be emitted first.
    pub fn before_tool_indicator(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::Text | LastOutput::Thinking);
        self.last = LastOutput::ToolIndicator;
        need_blank
    }

    /// Call before printing a thinking block. Returns true if a blank line
    /// should be emitted first.
    pub fn before_thinking(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::Text | LastOutput::ToolIndicator);
        self.last = LastOutput::Thinking;
        need_blank
    }

    /// Call after the todo list is rendered (it has its own trailing newline).
    pub fn after_todo_list(&mut self) {
        self.last = LastOutput::TodoList;
    }

    /// Call after newline_after_prompt is printed.
    pub fn after_prompt(&mut self) {
        self.last = LastOutput::Prompt;
    }
}

// ---------------------------------------------------------------------------
// Render mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderMode {
    #[default]
    Bat,
    Termimad,
    Raw,
}

impl std::fmt::Display for RenderMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderMode::Bat => write!(formatter, "bat"),
            RenderMode::Termimad => write!(formatter, "termimad"),
            RenderMode::Raw => write!(formatter, "raw"),
        }
    }
}

impl std::str::FromStr for RenderMode {
    type Err = String;

    fn from_str(string: &str) -> std::result::Result<Self, Self::Err> {
        match string.to_lowercase().as_str() {
            "bat" => Ok(RenderMode::Bat),
            "rich" | "termimad" => Ok(RenderMode::Termimad),
            "raw" => Ok(RenderMode::Raw),
            other => Err(format!(
                "unknown render mode '{}' (expected 'bat', 'termimad', or 'raw')",
                other
            )),
        }
    }
}

pub struct StreamingRenderer {
    buffer: String,
    skin: MadSkin,
    mode: RenderMode,
    pub(crate) started: bool,
    raw_table_lines: Vec<String>,
    code_block_lines: Vec<String>,
}

impl StreamingRenderer {
    pub fn new(mode: RenderMode) -> Self {
        Self {
            buffer: String::new(),
            skin: MadSkin::default_dark(),
            mode,
            started: false,
            raw_table_lines: Vec::new(),
            code_block_lines: Vec::new(),
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

        match self.mode {
            RenderMode::Bat => self.flush_bat(),
            RenderMode::Termimad => self.flush_termimad(),
            RenderMode::Raw => self.flush_raw(),
        }
    }

    pub fn finish(&mut self) -> io::Result<()> {
        match self.mode {
            RenderMode::Bat => {
                if !self.buffer.is_empty() {
                    let remaining = std::mem::take(&mut self.buffer);
                    let trimmed = remaining.trim_end_matches('\n');
                    let mut needs_newline = false;
                    for line in trimmed.lines() {
                        let is_fence = line.trim_start().starts_with("```");

                        if !self.code_block_lines.is_empty() {
                            self.code_block_lines.push(line.to_string());
                            if is_fence {
                                self.flush_bat_code_block()?;
                                needs_newline = false;
                            }
                        } else if is_fence {
                            self.flush_bat_table()?;
                            self.code_block_lines.push(line.to_string());
                            needs_newline = false;
                        } else if is_table_line(line) {
                            self.raw_table_lines.push(line.to_string());
                            needs_newline = false;
                        } else if line.is_empty() {
                            self.flush_bat_table()?;
                            println!();
                            needs_newline = false;
                        } else {
                            self.flush_bat_table()?;
                            print_with_bat(line);
                            needs_newline = true;
                        }
                    }
                    self.flush_bat_code_block()?;
                    self.flush_bat_table()?;
                    if needs_newline {
                        println!();
                    }
                }
            }
            RenderMode::Termimad => {
                if !self.buffer.is_empty() {
                    let remaining = std::mem::take(&mut self.buffer);
                    let trimmed = remaining.trim_end_matches('\n');
                    if !trimmed.is_empty() {
                        print!("{}", self.skin.term_text(trimmed));
                    }
                }
            }
            RenderMode::Raw => {
                if !self.buffer.is_empty() {
                    let remaining = std::mem::take(&mut self.buffer);
                    let trimmed = remaining.trim_end_matches('\n');
                    for line in trimmed.lines() {
                        if is_table_line(line) {
                            self.raw_table_lines.push(line.to_string());
                        } else {
                            self.flush_raw_table()?;
                            println!("{}", line);
                        }
                    }
                    self.flush_raw_table()?;
                }
            }
        }
        io::stdout().flush()
    }

    fn flush_bat(&mut self) -> io::Result<()> {
        self.buffer = normalize_spacing(&self.buffer);

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos].to_string();
            let is_fence = line.trim_start().starts_with("```");

            // If we're inside a code block, accumulate lines
            if !self.code_block_lines.is_empty() {
                self.buffer = self.buffer[newline_pos + 1..].to_string();
                self.code_block_lines.push(line);
                if is_fence {
                    self.flush_bat_code_block()?;
                }
                continue;
            }

            // Opening fence starts a new code block
            if is_fence {
                self.buffer = self.buffer[newline_pos + 1..].to_string();
                self.flush_bat_table()?;
                self.code_block_lines.push(line);
                continue;
            }

            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if is_table_line(&line) {
                self.raw_table_lines.push(line);
            } else {
                self.flush_bat_table()?;
                if line.is_empty() {
                    println!();
                } else {
                    print_with_bat(&format!("{}\n", line));
                }
                io::stdout().flush()?;
            }
        }
        Ok(())
    }

    fn flush_bat_code_block(&mut self) -> io::Result<()> {
        if self.code_block_lines.is_empty() {
            return Ok(());
        }

        let lines = std::mem::take(&mut self.code_block_lines);
        let block_text = lines.join("\n");
        print_with_bat(&block_text);
        println!();
        io::stdout().flush()
    }

    fn flush_bat_table(&mut self) -> io::Result<()> {
        if self.raw_table_lines.is_empty() {
            return Ok(());
        }

        let lines = std::mem::take(&mut self.raw_table_lines);
        let formatted = format_table(&lines);
        let table_text = formatted.join("\n");
        print_with_bat(&table_text);
        println!();
        io::stdout().flush()
    }

    fn flush_termimad(&mut self) -> io::Result<()> {
        self.buffer = normalize_spacing(&self.buffer);

        while let Some(boundary) = self.buffer.find("\n\n") {
            let complete = self.buffer[..boundary + 2].to_string();
            self.buffer = self.buffer[boundary + 2..].to_string();
            print!("{}", self.skin.term_text(&complete));
            io::stdout().flush()?;
        }

        if !self.in_code_block() && !self.in_table() {
            while let Some(newline_pos) = self.buffer.find('\n') {
                if newline_pos + 1 < self.buffer.len() || !self.buffer.ends_with('\n') {
                    let line = self.buffer[..newline_pos + 1].to_string();
                    self.buffer = self.buffer[newline_pos + 1..].to_string();
                    print!("{}", self.skin.term_text(&line));
                    io::stdout().flush()?;
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    fn flush_raw(&mut self) -> io::Result<()> {
        self.buffer = normalize_spacing(&self.buffer);

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos].to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if is_table_line(&line) {
                self.raw_table_lines.push(line);
            } else {
                self.flush_raw_table()?;
                println!("{}", line);
                io::stdout().flush()?;
            }
        }
        Ok(())
    }

    fn flush_raw_table(&mut self) -> io::Result<()> {
        if self.raw_table_lines.is_empty() {
            return Ok(());
        }

        let lines = std::mem::take(&mut self.raw_table_lines);
        let formatted = format_table(&lines);
        for line in &formatted {
            println!("{}", line);
        }
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

/// Ensure blank lines after markdown headers and tables when followed by
/// non-empty content. Skips content inside code fences.
fn normalize_spacing(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = Vec::with_capacity(lines.len());
    let mut in_fence = false;

    for (index, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }

        result.push(*line);

        if in_fence {
            continue;
        }

        let next_line = lines.get(index + 1);
        let next_is_non_empty = next_line.is_some_and(|next| !next.trim().is_empty());

        if !next_is_non_empty {
            continue;
        }

        let trimmed = line.trim_start();

        // Blank line after headers (e.g., `## Title`)
        let is_header = trimmed.starts_with('#')
            && trimmed
                .find(|character: char| character != '#')
                .is_some_and(|position| trimmed.as_bytes().get(position) == Some(&b' '));

        // Blank line after table rows when next line is clearly not a table row.
        // A line starting with `|` might be an incomplete table row from streaming,
        // so only treat lines NOT starting with `|` as table-ending.
        let is_table_end = is_table_line(line)
            && next_line.is_some_and(|next| !next.trim_start().starts_with('|'));

        if is_header || is_table_end {
            result.push("");
        }
    }

    // Preserve trailing newline if the original had one
    let mut output = result.join("\n");
    if text.ends_with('\n') {
        output.push('\n');
    }
    output
}

fn print_with_bat(text: &str) {
    if let Err(error) = bat::PrettyPrinter::new()
        .input_from_bytes(text.as_bytes())
        .language("markdown")
        .header(false)
        .line_numbers(false)
        .grid(false)
        .rule(false)
        .wrapping_mode(bat::WrappingMode::NoWrapping(false))
        .print()
    {
        tracing::debug!("bat rendering failed: {}", error);
    }
}

fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.ends_with('|') && trimmed.len() > 1
}

fn is_separator_row(cells: &[String]) -> bool {
    cells.iter().all(|cell| {
        cell.chars()
            .all(|character| character == '-' || character == ':')
    })
}

fn parse_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = trimmed
        .strip_prefix('|')
        .unwrap_or(trimmed)
        .strip_suffix('|')
        .unwrap_or(trimmed);
    inner
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn display_width(string: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(string)
}

fn format_table(lines: &[String]) -> Vec<String> {
    if lines.is_empty() {
        return Vec::new();
    }

    let parsed: Vec<Vec<String>> = lines.iter().map(|line| parse_table_row(line)).collect();

    let column_count = parsed.iter().map(|row| row.len()).max().unwrap_or(0);
    if column_count == 0 {
        return lines.to_vec();
    }

    let mut column_widths = vec![0usize; column_count];
    for row in &parsed {
        if is_separator_row(row) {
            continue;
        }
        for (column_index, cell) in row.iter().enumerate() {
            if column_index < column_count {
                column_widths[column_index] = column_widths[column_index].max(display_width(cell));
            }
        }
    }

    // Ensure minimum width of 3 for separator dashes
    for width in &mut column_widths {
        *width = (*width).max(3);
    }

    let mut result = Vec::new();
    for row in &parsed {
        if is_separator_row(row) {
            let separator: Vec<String> = column_widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect();
            result.push(format!("| {} |", separator.join(" | ")));
        } else {
            let padded: Vec<String> = (0..column_count)
                .map(|column_index| {
                    let cell = row.get(column_index).map(|s| s.as_str()).unwrap_or("");
                    let padding = column_widths[column_index].saturating_sub(display_width(cell));
                    format!("{}{}", cell, " ".repeat(padding))
                })
                .collect();
            result.push(format!("| {} |", padded.join(" | ")));
        }
    }

    result
}

pub fn render_tool_indicator(name: &str, input: &serde_json::Value) {
    let display_name = tool_display_name(name);
    let indicator = match tool_primary_param(name, input) {
        Some(value) => {
            let sanitized = value.replace('\n', " ");
            let truncated = truncate_display(&sanitized, 80);
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

pub fn render_thinking_block(thinking: &str, show_full: bool) {
    if show_full {
        eprintln!(
            "{}{}",
            "Thinking... ".with(Color::DarkGrey),
            thinking.with(Color::DarkGrey),
        );
    } else {
        let first_line = thinking.lines().next().unwrap_or("");
        let truncated = truncate_display(first_line, 80);
        eprintln!(
            "{}{}",
            "Thinking... ".with(Color::DarkGrey),
            truncated.with(Color::DarkGrey),
        );
    }
}

pub fn render_todo_list(items: &[crate::tools::todo::TodoItem]) {
    if items.is_empty() {
        return;
    }
    eprintln!();
    for item in items {
        let (marker, color) = match item.status.as_str() {
            "done" => ("[x]", Color::Green),
            "in_progress" => ("[~]", Color::Yellow),
            _ => ("[ ]", Color::DarkGrey),
        };
        eprintln!(
            "  {} {} {}",
            marker.with(color),
            item.id.clone().with(Color::White),
            item.description
        );
    }
    eprintln!();
}

pub fn tool_display_name_for_approval(name: &str) -> &str {
    tool_display_name(name)
}

pub fn tool_primary_param_for_approval<'a>(
    name: &str,
    input: &'a serde_json::Value,
) -> Option<&'a str> {
    tool_primary_param(name, input)
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
        "todo_write" => "TodoWrite",
        "spawn_agent" => "SpawnAgent",
        "scratchpad_write" => "ScratchpadWrite",
        "scratchpad_read" => "ScratchpadRead",
        "scratchpad_edit" => "ScratchpadEdit",
        "scratchpad_list" => "ScratchpadList",
        "scratchpad_delete" => "ScratchpadDelete",
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
        "spawn_agent" => "prompt",
        "scratchpad_write" | "scratchpad_read" | "scratchpad_edit" | "scratchpad_delete" => "name",
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
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("hello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_streaming_renderer_strips_leading_newlines() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("\n\nhello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_render_mode_default() {
        assert_eq!(RenderMode::default(), RenderMode::Bat);
    }

    #[test]
    fn test_is_table_line() {
        assert!(is_table_line("| A | B |"));
        assert!(is_table_line("|---|---|"));
        assert!(is_table_line("| single |"));
        assert!(!is_table_line("|"));
        assert!(!is_table_line("not a table"));
        assert!(!is_table_line("| no trailing pipe"));
    }

    #[test]
    fn test_parse_table_row() {
        let cells = parse_table_row("| Alpha | Beta | Gamma |");
        assert_eq!(cells, vec!["Alpha", "Beta", "Gamma"]);
    }

    #[test]
    fn test_parse_table_row_no_spaces() {
        let cells = parse_table_row("|A|B|C|");
        assert_eq!(cells, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_is_separator_row() {
        assert!(is_separator_row(&[
            "---".to_string(),
            "----".to_string(),
            "---".to_string()
        ]));
        assert!(is_separator_row(&[":--".to_string(), ":-:".to_string()]));
        assert!(!is_separator_row(&["Name".to_string(), "---".to_string()]));
    }

    #[test]
    fn test_format_table_alignment() {
        let lines = vec![
            "| Name | Value |".to_string(),
            "|------|-------|".to_string(),
            "| A | 100 |".to_string(),
            "| Beta | 2 |".to_string(),
        ];
        let result = format_table(&lines);
        assert_eq!(result.len(), 4);

        // All rows should have the same length
        let first_len = result[0].len();
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                row.len(),
                first_len,
                "row {} has length {} but expected {}",
                index,
                row.len(),
                first_len
            );
        }

        // Check content is padded
        assert_eq!(result[0], "| Name | Value |");
        assert_eq!(result[2], "| A    | 100   |");
        assert_eq!(result[3], "| Beta | 2     |");
    }

    #[test]
    fn test_format_table_wide_columns() {
        let lines = vec![
            "| # | Name | Type | Status | Score |".to_string(),
            "|---|------|------|--------|-------|".to_string(),
            "| 1 | Alpha | Primary | Pass | 98.5 |".to_string(),
            "| 2 | Beta | Secondary | Warn | 75.0 |".to_string(),
            "| 3 | Gamma | Primary | Pass | 91.2 |".to_string(),
        ];
        let result = format_table(&lines);
        let first_len = result[0].len();
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                row.len(),
                first_len,
                "row {} has length {} but expected {}",
                index,
                row.len(),
                first_len
            );
        }
    }

    #[test]
    fn test_format_table_empty() {
        let result = format_table(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_table_minimum_separator_width() {
        let lines = vec![
            "| A | B |".to_string(),
            "|---|---|".to_string(),
            "| C | D |".to_string(),
        ];
        let result = format_table(&lines);
        // Separator dashes should be at least 3 wide
        assert!(result[1].contains("---"));
    }

    #[test]
    fn test_format_table_emoji_single() {
        let lines = vec![
            "| Status | Name |".to_string(),
            "|---|---|".to_string(),
            "| 🟢 Pass | Alpha |".to_string(),
            "| 🔴 Fail | Beta |".to_string(),
        ];
        let result = format_table(&lines);
        assert_eq!(result.len(), 4);

        // All rows should have the same display width
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_format_table_emoji_multiple() {
        let lines = vec![
            "| Icon | Desc |".to_string(),
            "|---|---|".to_string(),
            "| 🟢🟢🟢 | Good |".to_string(),
            "| 🔴 | Bad |".to_string(),
        ];
        let result = format_table(&lines);
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_format_table_emoji_mixed_with_ascii() {
        let lines = vec![
            "| Segment | Change | Verdict |".to_string(),
            "|---|---|---|".to_string(),
            "| Canadian Banking | -9% | 🔴 Credit losses |".to_string(),
            "| Global Wealth | +17% | 🟢 AUM growth |".to_string(),
            "| Other | Flat | No emoji here |".to_string(),
        ];
        let result = format_table(&lines);
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_raw_mode_prints_text_verbatim() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.push_delta("**bold** text\n").unwrap();
        renderer.finish().unwrap();
        // Raw mode just prints text as-is; if it didn't panic, it works
    }

    #[test]
    fn test_raw_mode_table_buffering() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer
            .push_delta("| A | B |\n|---|---|\n| C | D |\n\nafter table\n")
            .unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_raw_mode_table_at_end() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer
            .push_delta("| A | B |\n|---|---|\n| C | D |")
            .unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_trims_trailing_newlines_raw() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.push_delta("hello\n\n\n").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_trims_trailing_newlines_rich() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("hello\n\n\n").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_only_newlines() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.started = true;
        renderer.buffer = "\n\n\n".to_string();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_normalize_spacing_adds_blank_line() {
        let input = "## Title\nBody text";
        let output = normalize_spacing(input);
        assert_eq!(output, "## Title\n\nBody text");
    }

    #[test]
    fn test_normalize_spacing_already_has_blank_line() {
        let input = "## Title\n\nBody text";
        let output = normalize_spacing(input);
        assert_eq!(output, "## Title\n\nBody text");
    }

    #[test]
    fn test_normalize_spacing_header_at_end() {
        let input = "## Title";
        let output = normalize_spacing(input);
        assert_eq!(output, "## Title");
    }

    #[test]
    fn test_normalize_spacing_inside_code_fence() {
        let input = "```\n## Not a header\ncode\n```";
        let output = normalize_spacing(input);
        assert_eq!(output, "```\n## Not a header\ncode\n```");
    }

    #[test]
    fn test_normalize_spacing_multiple_levels() {
        let input = "# H1\ntext\n### H3\nmore text";
        let output = normalize_spacing(input);
        assert_eq!(output, "# H1\n\ntext\n### H3\n\nmore text");
    }

    #[test]
    fn test_normalize_spacing_preserves_trailing_newline() {
        let input = "## Title\nBody\n";
        let output = normalize_spacing(input);
        assert_eq!(output, "## Title\n\nBody\n");
    }

    #[test]
    fn test_normalize_spacing_no_space_after_hash_is_not_header() {
        let input = "##not a header\ntext";
        let output = normalize_spacing(input);
        assert_eq!(output, "##not a header\ntext");
    }

    #[test]
    fn test_normalize_spacing_table_then_text() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |\n> blockquote";
        let output = normalize_spacing(input);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote");
    }

    #[test]
    fn test_normalize_spacing_table_already_has_blank_line() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote";
        let output = normalize_spacing(input);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote");
    }

    #[test]
    fn test_normalize_spacing_table_inside_code_fence() {
        let input = "```\n| A | B |\n| 1 | 2 |\ncode\n```";
        let output = normalize_spacing(input);
        assert_eq!(output, "```\n| A | B |\n| 1 | 2 |\ncode\n```");
    }

    #[test]
    fn test_normalize_spacing_table_at_end() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |";
        let output = normalize_spacing(input);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |");
    }
}
