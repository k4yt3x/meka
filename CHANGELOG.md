# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-04-04

### Changed

- Optimize prompt caching to avoid unnecessary KV cache invalidation across turns and tool-use loops.

## [0.7.0] - 2026-04-04

### Changed

- Adapted Claude provider to match current claude-code header and attestation requirements.

## [0.6.1] - 2026-03-28

### Fixed

- Fixed build failure with rmcp 1.3.0 by using `OAuthClientConfig` builder API.
- OpenAI provider not parsing top-level `name`/`arguments` in proxy tool call responses.

## [0.6.0] - 2026-03-25

### Added

- Shift+Enter as an alternative to Alt+Enter for inserting newlines in the REPL.
- `headers` parameter for `fetch_url` and `web_search` tools to set custom HTTP headers.
- `regex` parameter for `fetch_url` tool to filter page content by pattern.

### Changed

- Changed default web user agent to Chrome for better content fetching success rates.

## [0.5.3] - 2026-03-18

### Fixed

- User prompts are no longer recorded in history when the server returns an error.
- The blank line after the agent's response is now printed even when an error occurs.
- Partial assistant responses are now saved to history on Ctrl+C interrupt.

### Changed

- Reduced `fetch_url` default `max_length` from 50000 to 30000.

## [0.5.2] - 2026-03-17

### Changed

- `fetch_url` tool accepts optional `max_length` parameter (default: 50000, 0 for no limit).

## [0.5.1] - 2026-03-17

### Fixed

- Claude OAuth requests failing with 400.
- `urldecode` incorrectly handling multi-byte UTF-8 percent-encoded sequences.

### Changed

- Generate dynamic billing header with content-based hashing for Claude OAuth requests.
- Replaced custom HTML search result parsers with `scraper` crate for CSS selectors.
- Replaced custom `urldecode` with `percent-encoding` crate (already a transitive dep).
- Replaced custom `ceil_char_boundary` utility with stdlib `str::ceil_char_boundary`.
- Reuse a single `reqwest::Client` for web tools instead of constructing one per request.
- Extracted duplicated timestamp calculation in Claude provider into a helper function.

## [0.5.0] - 2026-03-16

### Added

- OAuth auth for MCP HTTP servers: client credentials, JWT signing, and PKCE.
- Persistent MCP OAuth credential storage in SQLite with automatic token refresh.

### Changed

- Default render mode changed from `raw` to `rich`.
- Raw render mode now prints output verbatim, only formatting tables with aligned columns.
- Upgraded `reqwest` from 0.12 to 0.13.

### Removed

- Custom raw mode ANSI markdown renderer (replaced with passthrough + table alignment).
- `unicode-width` direct dependency.

### Fixed

- Trailing newlines in agent responses causing duplicate blank lines before the next prompt.

## [0.4.1] - 2026-03-14

### Added

- `display.show_path_in_prompt` config to toggle working directory in the prompt.

## [0.4.0] - 2026-03-14

### Added

- Working directory displayed in the shell prompt with tilde shortening for home dir.
- `/cd` slash command for changing the working directory.
- MCP client support: external tool servers via `[[mcp.servers]]` with stdio and HTTP.
- MCP tools namespaced as `server__tool` with per-server permission configuration.
- `delete` subcommand to delete specific or all sessions.
- `list` subcommand to display past sessions with timestamps and preview text.
- `export` subcommand to export session history as Markdown.
- Raw markdown render mode with ANSI highlighting via `--render-mode raw` or config.
- Table column alignment in raw render mode with Unicode-width-aware CJK padding.
- `Database` error variant for SQLite errors (previously misclassified as `Config`).
- Unit tests for CLI parsing, slash commands, PKCE/OAuth, and rendering (31 tests).
- Unit tests for malformed API response handling (missing `id`, `name`, `message`).

### Changed

- Default render mode changed from `rich` to `raw`.
- Split `display.show_session_id` into `on_create` and `on_exit` variants.
- Replaced all `.expect()` calls in production code with error propagation via `?`.
- Replaced all `let _ =` on fallible operations with proper error logging.
- Removed organizational section divider comments to comply with coding guidelines.
- Deduplicated stop reason parsing into `parse_openai/claude_stop_reason` helpers.
- Deduplicated OpenAI streaming tool call finalization into a helper function.
- Config file writing now uses proper TOML serialization instead of string formatting.
- Replaced `unwrap_or_default()` in message serialization with error propagation.
- Added `tracing::warn!` for fallback JSON parsing of malformed tool arguments.
- Introduced `AgentOptions` struct to reduce `Agent::new` parameter count.
- Resolved all clippy warnings (collapsible if, wildcard patterns, C string literals).
- Renamed single-letter closure variables in provider parsing to descriptive names.
- Replaced `unwrap_or_default()` on tool call fields with proper error propagation.
- Replaced direct JSON indexing with `.get()` and error handling in provider parsing.
- Split `provider.rs` into module: shared types, `claude.rs`, and `openai.rs`.
- Split `tools.rs` into module: registry, `file.rs`, `search.rs`, `shell.rs`, `web.rs`.

### Fixed

- Streaming mode now shows full API error body instead of a generic error message.
- Multi-line paste now inserts all lines into the buffer instead of executing immediately.
- TOML injection in `write_config_file` when API keys contain special characters.
- Pre-existing test compilation errors in `ClaudeProvider::new` and `create_provider`.

## [0.3.1] - 2026-03-12

### Fixed

- OAuth token refresh failing with 400 due to missing `client_id` and form-encoded body.

## [0.3.0] - 2026-03-11

### Added

- First-launch setup wizard for provider, authentication, and model configuration.
- `agsh setup` subcommand to re-run the configuration wizard.
- OAuth Authorization Code flow with PKCE for Claude provider authentication.
- OAuth token auth for Claude via `CLAUDE_OAUTH_TOKEN` env var or config.
- Database-backed OAuth token storage with automatic refresh.
- Configurable OAuth token refresh endpoint via `provider.oauth_token_url`.

### Changed

- Renamed `anthropic` provider to `claude` (breaking: `--provider anthropic` removed).
- Renamed `ANTHROPIC_API_KEY` env var to `CLAUDE_API_KEY`.
- API key no longer required at startup when an OAuth token is stored in the database.

## [0.2.0] - 2026-03-06

### Added

- Slash commands: `/help`, `/exit`, `/clear`, `/session`, `/permission`, `/compact`.
- Skills: user-defined Markdown knowledge files the agent can discover and read.
- Configurable context window limiting via `[session] context_messages`.
- Automatic session cleanup via `[session] retention_days` and `max_storage_bytes`.

### Changed

- One-shot prompt is now a positional argument (`agsh "prompt"`) instead of a flag.
- Switched `reqwest` from `native-tls` to `rustls-tls` for pure-Rust TLS.
- Added release profile optimizations (`lto`, `codegen-units = 1`, `strip`).
- Added Rust dependency caching in CI workflow.
- Removed OpenSSL system dependency installation from CI.

## [0.1.2] - 2026-03-05

### Added

- Windows binary icon embedding via `winres`.

### Fixed

- Panic on multi-byte UTF-8 chars in web search HTML parsers (Google, Bing).

## [0.1.1] - 2026-03-05

### Added

- Read-only filesystem sandboxing for shell commands using Landlock and sandbox-exec.
- Configurable sandbox toggle via `[shell] sandbox` config option.
- Conditional system prompt for read mode based on sandbox availability.

### Fixed

- Panic on multi-byte UTF-8 chars in web search results and URL fetching truncation.

## [0.1.0] - 2026-03-05

### Added

- Interactive REPL shell with natural language input.
- One-shot mode via positional `[PROMPT]` argument.
- OpenAI and Claude LLM provider support with streaming.
- Three-level permission system (none/read/write) with Shift+Tab cycling.
- Built-in tools: `read_file`, `write_file`, `edit_file`, `find_files`, and more.
- Session persistence with SQLite (create, resume with `-s`, continue with `-c`).
- Session locking to prevent concurrent access.
- `!` prefix shell escape for direct command execution.
- `exit`/`quit` keywords and Ctrl+D to leave the shell.
- TOML configuration file with `[provider]`, `[display]`, and `[web]` sections.
- Configurable user agent for web requests via `[web] user_agent`.
- Cross-platform support for Windows (PowerShell) and macOS.
- Platform-specific OS detection in system prompt (Linux, macOS, Windows).
- Leading newline stripping from LLM streaming output.
- mdBook documentation site.
- GitHub Actions workflows for documentation deployment and release builds.
- MIT license.
