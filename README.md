# agsh

An agentic shell where you speak human, not bash.

![agsh Screenshot](https://github.com/user-attachments/assets/e614ed1f-4fa3-45ef-b194-fd5c9ec554cd)

## Overview

agsh (agentic shell) is an interactive shell that replaces traditional command syntax with natural language. Describe what you want, and the agent reads files, searches code, runs commands, and browses the web to get it done. One binary, one config, works with any OpenAI-compatible or Anthropic API.

## Installation

Download a pre-built binary from [GitHub Releases](https://github.com/k4yt3x/agsh/releases/latest), or install with Cargo:

```bash
cargo install --git https://github.com/k4yt3x/agsh.git
```

## Quick Start

1. Create `~/.config/agsh/config.toml` and configure the provider and model you want to use. For example, to use OpenRouter with an Anthropic model:

```toml
[provider]
name = "openai"
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-or-v1-..."
model = "anthropic/claude-opus-4.6"
```

2. Run `agsh` and start typing. Press Shift+Tab to cycle permissions (none, read, write):

```
agsh [r] > find all TODO comments in this project
agsh [w] > install and start nginx
```

See the [documentation](https://k4yt3x.github.io/agsh) for the full usage guide.

## Tools

The agent has access to the following built-in tools:

- **Shell**: execute commands and read their output
- **ReadFile / WriteFile / EditFile**: read, create, and modify files
- **FindFiles**: find files by name or glob pattern
- **SearchContents**: search file contents with regex (powered by ripgrep)
- **FetchUrl**: fetch and read web page content
- **WebSearch**: search the web for up-to-date information

## Permissions

The prompt indicator shows the current permission mode. Press **Shift+Tab** to cycle between modes:

- `[n]` **none**: no tools available, text-only responses
- `[r]` **read**: read-only tools (file reading, searching, web); cannot modify anything
- `[w]` **write**: all tools enabled, including shell execution and file writes

## Sessions

Conversations are persisted in a local SQLite database and can be resumed:

- `agsh -c` continues the last session
- `agsh -s <UUID>` resumes a specific session by ID

## Shell Escape

Prefix input with `!` to execute a command directly, bypassing the LLM:

```
agsh [r] > !uname -a
agsh [r] > !docker ps
```

Type `exit`, `quit`, or press **Ctrl+D** to leave the shell.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).\
Copyright 2026 K4YT3X.
