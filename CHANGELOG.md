# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Configurable context window limiting via `[session] context_messages` to cap messages sent to the LLM API
- Automatic session cleanup via `[session] retention_days` (time-based) and `[session] max_storage_bytes` (size-based)

### Changed

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
- One-shot mode via `-p`/`--prompt` flag
- OpenAI and Anthropic LLM provider support with streaming
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
