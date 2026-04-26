# agsh

A general-purpose AI agent runtime.

> [!CAUTION]
> Agents can perform potentially destructive actions. Exercise caution when granting write permissions. It is not recommended to run agsh on important systems with write permissions enabled.

> [!WARNING]
> Agents can consume a large number of tokens on complex tasks. If you're on a subscription, be prepared for quota exhaustion; if you are billed per token, it is recommended that you set a spending limit on the API key.

![agsh Screenshot](https://github.com/user-attachments/assets/e94c40ee-76ae-4b00-bcfe-1c1d9d075a2b)

## Overview

agsh is a general-purpose AI agent runtime that provides LLMs with a rich set of tools — web search, shell execution, file editing, and more — to accomplish complex tasks. Use it as a natural-language shell, a system diagnostic helper, a research or data-analysis assistant, for general Q&A, or to add agentic capabilities to other applications.

Supported providers:

- **OpenAI API** — bring your own API key. Also works with any OpenAI-compatible endpoint.
- **OpenAI Codex** — authenticate with a ChatGPT subscription.
- **Claude API** — bring your own API key.
- **Claude OAuth** — authenticate with a Claude subscription.

## Installation

Download a pre-built binary from [GitHub Releases](https://github.com/k4yt3x/agsh/releases/latest), or install with Cargo:

```bash
cargo install --locked --git https://github.com/k4yt3x/agsh.git
```

## Quick Start

Run `agsh setup` for an interactive wizard that picks a provider, runs the OAuth login if applicable, and writes the config for you.

To configure manually, create `~/.config/agsh/config.toml`. For example, to use OpenRouter with a Claude model:

```toml
[provider]
name = "openai-api"
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-or-v1-..."
model = "anthropic/claude-opus-4.6"
```

Run `agsh` and start typing. Press Shift+Tab to cycle permissions (none, read, ask, write):

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
- **Scratchpad**: session-scoped working memory (write, read, edit, list, delete)
- **TodoWrite**: structured task tracking for multi-step work
- **SpawnAgent**: delegate research tasks to a read-only sub-agent
- **Skill**: load reusable prompt templates on demand
- **RenderImage**: render an image into the conversation for vision-capable models
- **MCP resources / prompts**: read or render content from configured MCP servers

Long-output tools support an optional `scratchpad` parameter to save output directly to the scratchpad.

## Permissions

The prompt indicator shows the current permission mode. Press **Shift+Tab** to cycle between modes:

- `[n]` **none**: no tools available, text-only responses
- `[r]` **read**: read-only tools (file reading, searching, web, sandboxed shell); cannot modify anything
- `[a]` **ask**: all tools available, but each call requires user approval
- `[w]` **write**: all tools enabled, including shell execution and file writes

## Sessions

Conversations are persisted in a local SQLite database and can be resumed:

- `agsh -c` continues the last session
- `agsh -c <UUID>` resumes a specific session by ID
- `agsh list` lists past sessions
- `agsh export <UUID>` exports a session as Markdown
- `/export` exports the current session from within the shell
- `/compact` summarizes and compacts the session history

## Features

- **Extended/adaptive thinking**: enabled by default for Claude models that support it
- **Syntax-highlighted output**: bat-powered markdown rendering with code block highlighting
- **Auto-compact**: automatically compacts the conversation when approaching the context limit
- **MCP support**: extend the agent with tools from external MCP servers
- **Skills**: load reusable prompt templates from `~/.config/agsh/skills/`

## Shell Escape

Prefix input with `!` to execute a command directly, bypassing the LLM:

```
agsh [r] > !uname -a
agsh [r] > !docker ps
```

Type `exit`, `quit`, or press **Ctrl+D** to leave the shell.

## AI Use Declaration

AI tools were used to assist the design and implementation of this project. All design decisions were made by humans, and every change was reviewed and approved by a human maintainer.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).\
Copyright 2026 K4YT3X.
