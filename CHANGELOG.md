# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-11

### Added

- First-launch setup wizard that guides new users through provider, authentication, and model configuration
- `agsh setup` subcommand to re-run the configuration wizard
- OAuth Authorization Code flow with PKCE for Claude provider authentication
- OAuth token authentication for the Claude provider via `CLAUDE_OAUTH_TOKEN` env var or `provider.oauth_token` config
- Database-backed OAuth token storage with automatic refresh
- Configurable OAuth token refresh endpoint via `provider.oauth_token_url`

### Changed

- Renamed `anthropic` provider to `claude` (breaking: `--provider anthropic` no longer works)
- Renamed `ANTHROPIC_API_KEY` env var to `CLAUDE_API_KEY`
- API key is no longer required at startup when an OAuth token is stored in the database

## [0.2.0] - 2026-03-06

### Added

- Slash commands: `/help`, `/exit`, `/clear`, `/session`, `/permission`, and `/compact` for in-shell control
- Skills: user-defined Markdown knowledge files in `~/.config/agsh/skills/` that the agent can discover and read on demand
- Configurable context window limiting via `[session] context_messages` to cap messages sent to the LLM API
- Automatic session cleanup via `[session] retention_days` (time-based) and `[session] max_storage_bytes` (size-based)

### Changed

- One-shot prompt is now a positional argument (`agsh "prompt"`) instead of a flag (`agsh -p "prompt"`)
- Switched `reqwest` from `native-tls` (OpenSSL) to `rustls-tls` for pure-Rust TLS, eliminating C compilation dependency
- Added release profile optimizations (`lto`, `codegen-units = 1`, `strip`)
- Added Rust dependency caching in CI workflow
- Removed OpenSSL system dependency installation from CI

## [0.1.2] - 2026-03-05

### Added

- Windows binary icon embedding via `winres`

### Fixed

- Panic on multi-byte UTF-8 characters in web search HTML parsers (extract_href, Google, Bing)

## [0.1.1] - 2026-03-05

### Added

- Read-only filesystem sandboxing for shell commands in read mode using Landlock (Linux) and sandbox-exec (macOS)
- Configurable sandbox toggle via `[shell] sandbox` config option
- Conditional system prompt for read mode based on sandbox availability

### Fixed

- Panic on multi-byte UTF-8 characters in web search results and URL fetching truncation

## [0.1.0] - 2026-03-05

### Added

- Interactive REPL shell with natural language input
- One-shot mode via positional `[PROMPT]` argument
- OpenAI and Claude LLM provider support with streaming
- Three-level permission system (none/read/write) with Shift+Tab cycling
- Built-in tools: `read_file`, `write_file`, `edit_file`, `find_files`, `search_contents`, `execute_command`, `fetch_url`, `web_search`
- Session persistence with SQLite (create, resume by UUID with `-s`, continue last with `-c`)
- Session locking to prevent concurrent access
- `!` prefix shell escape for direct command execution
- `exit`/`quit` keywords and Ctrl+D to leave the shell
- TOML configuration file with `[provider]`, `[display]`, and `[web]` sections
- Configurable user agent for web requests via `[web] user_agent`
- Cross-platform support for Windows (PowerShell) and macOS
- Platform-specific OS detection in system prompt (Linux, macOS, Windows)
- Leading newline stripping from LLM streaming output
- mdBook documentation site
- GitHub Actions workflows for documentation deployment and release builds
- MIT license
