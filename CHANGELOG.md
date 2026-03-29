# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fixed build failure with rmcp 1.3.0 by using `OAuthClientConfig` builder API instead of struct literal construction
- OpenAI provider now handles flattened tool call responses from API proxies

## [0.6.0] - 2026-03-25

### Added

- Shift+Enter as an alternative to Alt+Enter for inserting newlines in the REPL
- `headers` parameter for `fetch_url` and `web_search` tools to set custom HTTP headers
- `regex` parameter for `fetch_url` tool to filter page content by pattern

### Changed

- Changed default web user agent to Chrome for better content fetching success rates

## [0.5.3] - 2026-03-18

### Fixed

- User prompts are no longer recorded in conversation history when the server returns an error
- The blank line after the agent's response is now printed even when an error occurs
- Partial assistant responses are now saved to conversation history when the user interrupts with Ctrl+C

### Changed

- Reduced `fetch_url` default `max_length` from 50000 to 30000

## [0.5.2] - 2026-03-17

### Changed

- `fetch_url` tool accepts optional `max_length` parameter (default: 50000, set to 0 for no limit)

## [0.5.1] - 2026-03-17

### Fixed

- Claude OAuth requests failing with 400
- `urldecode` incorrectly handling multi-byte UTF-8 percent-encoded sequences (e.g., `%C3%A9` producing mojibake instead of 'é')

### Changed

- Generate dynamic billing header with content-based hashing for Claude OAuth requests
- Replaced custom HTML search result parsers with `scraper` crate for CSS selector-based extraction
- Replaced custom `urldecode` implementation with `percent-encoding` crate (already a transitive dependency)
- Replaced custom `ceil_char_boundary` utility with stdlib `str::ceil_char_boundary`
- Reuse a single `reqwest::Client` for web tools instead of constructing one per request
- Extracted duplicated timestamp calculation in Claude provider into a helper function

## [0.5.0] - 2026-03-16

### Added

- OAuth authentication for MCP HTTP servers: client credentials, client credentials with JWT signing, and OAuth 2.1 authorization code flow with PKCE
- Persistent MCP OAuth credential storage in SQLite with automatic token refresh via rmcp's `AuthorizationManager`

### Changed

- Default render mode changed from `raw` to `rich`
- Raw render mode now prints agent output verbatim, only formatting tables with aligned columns
- Upgraded `reqwest` from 0.12 to 0.13

### Removed

- Custom raw mode ANSI markdown renderer (replaced with passthrough + table alignment)
- `unicode-width` direct dependency

### Fixed

- Trailing newlines in agent responses causing duplicate blank lines before the next prompt

## [0.4.1] - 2026-03-14

### Added

- `display.show_path_in_prompt` config option to toggle working directory display in the prompt (default: `true`)

## [0.4.0] - 2026-03-14

### Added

- Working directory displayed in the shell prompt with tilde shortening for home directory
- `/cd` slash command for changing the working directory (affects prompt, agent context, and all tool operations)
- MCP (Model Context Protocol) client support: configure external tool servers via `[[mcp.servers]]` in config file with stdio and streamable HTTP transports
- MCP tools namespaced as `server__tool` and registered alongside built-in tools with per-server permission configuration
- `delete` subcommand (`agsh delete <id>...` or `agsh delete --all`) to delete specific or all sessions
- `list` subcommand (`agsh list [-n <count>]`) to display past sessions with timestamps and preview text
- `export` subcommand (`agsh export <session-id> [-o <path>]`) to export session history as Markdown (default: `session-<id>.md`, `-o -` for stdout)
- Raw markdown render mode (`--render-mode raw` CLI flag or `display.render_mode = "raw"` config option) that outputs markdown with ANSI color/style highlighting instead of rich terminal formatting
- Table column alignment in raw render mode with Unicode-width-aware padding for CJK characters
- `Database` error variant for SQLite-related errors (previously misclassified as `Config` errors)
- Unit tests for CLI argument parsing, slash command parsing, PKCE/OAuth helpers, and rendering utilities (31 new tests)
- Unit tests for malformed API response handling (missing tool call `id`, `name`, and `message` fields)

### Changed

- Default render mode changed from `rich` to `raw`
- Split `display.show_session_id` into `display.show_session_id_on_create` (default: false) and `display.show_session_id_on_exit` (default: true) for independent control

- Replaced all `.expect()` calls in production code with proper error propagation via `?`
- Replaced all `let _ =` on fallible operations with proper error logging
- Removed organizational section divider comments to comply with coding guidelines
- Deduplicated stop reason parsing into `parse_openai_stop_reason` and `parse_claude_stop_reason` helpers
- Deduplicated OpenAI streaming tool call finalization into `finalize_tool_call_accumulators` helper
- Config file writing now uses proper TOML serialization instead of string formatting
- Replaced `unwrap_or_default()` in message serialization with error propagation
- Added `tracing::warn!` for fallback JSON parsing of malformed tool arguments
- Introduced `AgentOptions` struct to reduce `Agent::new` parameter count
- Resolved all clippy warnings (collapsible if, wildcard in or-patterns, manual C string literals, needless lifetimes, etc.)
- Renamed single-letter closure variables in provider parsing to use full descriptive names
- Replaced `unwrap_or_default()` on required tool call fields (`id`, `name`) with proper error propagation
- Replaced direct JSON indexing (`&value["key"]`) with `.get()` and proper error handling in provider response parsing
- Split `provider.rs` (1,840 lines) into a module: `provider.rs` (shared types/trait), `provider/claude.rs`, `provider/openai.rs`
- Split `tools.rs` (1,531 lines) into a module: `tools.rs` (trait/registry), `tools/file.rs`, `tools/search.rs`, `tools/shell.rs`, `tools/web.rs`, `tools/util.rs`

### Fixed

- Streaming mode now shows full API error body (e.g., rate limit details, reset times) instead of a generic error
- Multi-line paste now inserts all lines into the buffer instead of executing the first line immediately
- TOML injection vulnerability in `write_config_file` when API keys contain special characters
- Pre-existing test compilation errors in `ClaudeProvider::new` and `create_provider` calls (missing argument)

## [0.3.1] - 2026-03-12

### Fixed

- OAuth token refresh failing with 400 due to missing `client_id` parameter and form-encoded body

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
