# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `[display].input_style = "reverse"` uses ANSI reverse video (swaps terminal fg/bg).
- `[mcp].strict`, `grace_seconds`, `connect_timeout_seconds` tune the per-turn readiness gate.
- `[[mcp.servers]].disabled` skips a server at startup without removing it from config.
- `agsh mcp disable <name>` / `agsh mcp enable <name>` toggle the disabled flag in config.toml.
- `agsh mcp add --disabled` stages a server without connecting to it on the next start.
- `web_search` detects DuckDuckGo CAPTCHA pages and returns a clear error instead of silent empty.
- `[web]` gains reqwest knobs: request/connect/read timeouts, max redirects, proxy, CA bundle, TLS.

### Changed

- MCP servers connect in parallel in the background; REPL opens immediately, not after Σ(connect).
- Default strict gate: turns abort when any enabled MCP server isn't connected.
- `/mcp list` in the REPL shows live state (connected / pending / failed / disabled) per server.
- `web_search` output: normalized whitespace, source-domain line, bold markdown on matched terms.

### Removed

- `web_search` Google and Bing engines (both consistently bot-blocked).

## [0.14.0] - 2026-04-20

### Added

- `[tools]` config: `allowed_tools`, `disabled_tools`, and `tool_permissions` filters for built-in tools.
- `agsh tools list` prints every built-in tool with its effective permission and enabled state.
- `[display].input_style` styles REPL input so submitted prompts stand out in scrollback.

## [0.13.1] - 2026-04-20

### Changed

- `agsh mcp tools --help` description trimmed to a single line.
- Renamed `src/shell.rs` → `src/repl.rs` and `src/mcp/env.rs` → `src/mcp/expand.rs` for clearer module names.

## [0.13.0] - 2026-04-19

### Added

- `AGSH_CONFIG_DIR` env var overrides the default config directory on every platform.
- System prompt now lists every registered tool with its required permission level inline.
- Per-turn user message carries a `[Permission context]` block naming the current level.
- Per-tool MCP permission chain: `tool_permissions` > `permission` > `readOnlyHint` > `default_permission`.
- `[mcp] default_permission` config key: global fallback when no server/tool/hint applies.
- `[[mcp.servers]]` supports `allowed_tools` / `disabled_tools` / `tool_permissions` overrides.
- `agsh mcp add` flags: `--allow-tool`, `--disable-tool`, `--tool-permission NAME=LEVEL` (repeatable).
- `agsh mcp get <name>` now lists allow/block lists and per-tool permission overrides.
- Stale entries in `allowed_tools`/`disabled_tools`/`tool_permissions` emit a `warn!` at connect time.
- `agsh mcp tools <name>` lists every advertised tool with resolved permission and which chain step won.
- `agsh mcp` CLI: `list`, `get`, `add`, `remove`, `reconnect`, `login`, `logout` subcommands.
- `agsh mcp add <name> <url-or-command> [args]` auto-detects transport (URL → http, else stdio).
- `agsh mcp add` flags for env/headers, permission, auth (oauth, client-credentials, -jwt, token).
- `agsh mcp add` probes HTTP servers post-persist (RFC 6750 / RFC 9728): 3 s redirects-off GET.
- `agsh mcp add` auto-runs OAuth on auth-required / `--auth oauth`; `--no-login` skips.
- `agsh mcp add` auto-login failure or Ctrl-C rolls the entry back (config + creds + probe cache).
- `agsh mcp login <name>` assumes OAuth authorization_code on HTTP servers without an `[auth]` block.
- OAuth callback races the bound TCP listener against a stdin paste so logins work over SSH.
- `/mcp login <server>` and `/mcp logout <server>` REPL commands mirror the CLI subcommands.
- Server `InitializeResult.instructions` spliced into the system prompt each turn.
- Progress notifications forwarded to the REPL as a live status line under the running tool call.
- Form + URL elicitation — the shell prompts the user and returns typed values to the server.
- Tool annotations / `_meta` / `structuredContent` preserved through to the provider.
- Builtin MCP resource/prompt tools for list/read, subscribe/unsubscribe, and get-prompt flows.
- OAuth token revocation via `agsh mcp logout` (RFC 7009) + 15-min auth-probe cache for 401s.
- Tool-call timeout (`AGSH_MCP_TOOL_TIMEOUT`, default 600s) with best-effort cancellation.
- Exponential-backoff reconnect for HTTP MCP (5 attempts, 1s → 30s); stdio retries once.
- `${VAR}` / `${VAR:-default}` expansion across MCP command, args, env, url, headers, auth_token.
- `headers_helper` config field: per-server script emits dynamic HTTP headers at connect-time.
- Windows stdio: auto-wrap `npx`, `.cmd`, `.bat`, `.ps1` commands in `cmd /c`.
- Unicode + server-name sanitisation of MCP strings; `agsh`, `ide`, `mcp_*` names rejected.
- `sampling/createMessage` server-to-client flow, opt-in via `sampling = true` + `sampling_limit`.
- `roots/list` advertises the agsh current working directory.
- MCP image tool-result content reaches providers as image blocks instead of `[image content]`.
- OAuth callback listener binds to an ephemeral port when `redirect_port` is omitted.
- Ctrl-C now sends `notifications/cancelled` to the server with the in-flight request id.
- Dynamic tool list refresh on `tools/list_changed` — new tools picked up without restart.

### Changed

- `execute_command` description names the shell per platform and warns against double-PowerShell wrapping.
- Per-turn `[Permission context]` is a constant two-line block; no longer enumerates blocked tools.
- System prompt tool catalogue is leaner: name + permission for active tools, short summaries for deferred.
- System prompt and `body["tools"]` no longer depend on permission level; toggles keep the cache warm.
- **Breaking**: MCP tools with no `readOnlyHint` and no `[mcp].default_permission` now require `Write`.

### Security

- MCP progress + elicitation strings sanitised before reaching the terminal; no ANSI/RTL spoofing.
- MCP tool-result images capped at 10 MiB and restricted to PNG/JPEG/GIF/WebP; else a placeholder.
- MCP sampling `system_prompt` stripped of Cc/Cf codepoints before reaching the provider.
- `read_mcp_resource` + `get_mcp_prompt` + list tools sanitise server-supplied text and URIs.
- `read_mcp_resource` total output capped at 10 MiB; oversized chunks replaced with a marker.
- `headers_helper` stdout capped at 64 KiB, stderr at 4 KiB, to contain helper misbehaviour.
- OAuth revocation rejects redirects, caps metadata at 256 KiB, pins endpoint to issuer origin.
- OAuth callback `error=…` query parameter is stripped of Cc/Cf codepoints before display.
- JWT signing key files rejected on Unix when group/other perm bits are set (must be 0600).
- MCP cancellation notifications now time out after 2 s so a hung transport can't stall Ctrl-C.
- `agsh mcp add`/`remove` writes config.toml atomically and chmods it 0600 (dir 0700) on Unix.
- `agsh mcp add` propagates config-read errors instead of silently treating them as an empty file.

### Fixed

- `${VAR}` expansion for MCP config preserves multi-byte UTF-8 (previously corrupted non-ASCII).
- MCP tools with an unserializable input schema are skipped with a warning.
- OAuth-authenticated MCP transports now reconnect cleanly mid-session.
- MCP `sampling/createMessage` has a 60 s provider timeout and refunds the sampling slot on error.
- `agsh mcp remove` now clears that server's entries from the resource-update ledger.
- `agsh mcp remove` now also best-effort revokes stored OAuth tokens at the provider (RFC 7009).
- MCP auth-probe cache with `ttl = 0` now correctly treats every entry as stale.
- rmcp's SSE-reconnect warning floored at `error` in default filter; CDN idle resets no longer spam.

## [0.12.0] - 2026-04-18

### Security

- File tools route I/O through the canonical path with `O_NOFOLLOW` on Unix, closing a symlink-swap TOCTOU.
- `fetch_url` caps response body at 10 MiB to defend against gzip/brotli decompression bombs.
- Session data dir, lock dir, and DB file are created 0700/0700/0600 on Unix regardless of umask.
- Tool calls with unparseable JSON arguments are now rejected instead of silently run with `{}`.
- Windows Low-integrity sandbox scrubs the child environment so provider API keys aren't inherited.
- `execute_command` on Unix kills the whole process group on timeout so grandchildren can't outlive it.
- LLM-supplied regex patterns are compiled with 1 MiB size/DFA limits to bound compile-time memory.
- Tool indicators strip ANSI CSI escapes and C0 controls so commands can't spoof the permission prompt.
- Permission enforcement now reads the shared permission atomically at the dispatch site.

### Added

- `tests/cli.rs` end-to-end smoke tests for `--version`, `--help`, unknown flags.
- `render::render_error` and `render::render_provider_setup_hint` helpers for CLI output.
- Module-level `//!` doc comments across the codebase; CI runs `cargo doc -D warnings`.
- CI test job runs on Linux, macOS, and Windows to cover platform-specific sandbox code.
- Windows `execute_command` sandbox via Low-integrity token with handle-list inheritance filter.
- Windows sandbox falls back to `CreateProcessWithTokenW` when `SE_INCREASE_QUOTA_NAME` is missing.

### Changed

- Session locking uses OS file locks via `fd-lock` so kernel-released locks survive hard kills.
- `SessionManager::lock_session` is now sync and returns a `SessionLock` RAII handle.
- Schema migration drops the legacy `sessions.locked_by` column to unstick old sessions.
- `execute_command` on Windows invokes PowerShell with `-NoProfile -NonInteractive` always.
- `execute_command` children no longer inherit the agent's stdin; they see immediate EOF.

### Fixed

- `default_database_path` falls back to `$HOME/.local/share` and errors cleanly when unset.
- Stuck sessions from PID-based locking surviving hard kills (resolved via OS file locks).
- Windows sandbox normal-exit drain now times out after 5s instead of hanging on a grandchild.

## [0.11.0] - 2026-04-17

### Added

- `skill` tool for loading named skills.
- YAML frontmatter for skills (description, when_to_use, allowed_tools, version, user_invocable).
- `${AGSH_SKILL_DIR}` and `${AGSH_SESSION_ID}` substitution in skill bodies.
- `[prompt] instructions` config for system-wide instructions injected into every session's prompt.
- `fetch_url` returns a multimodal Image block for image URLs (sandboxed mode, no disk I/O).
- `fetch_url` and `read_file` convert TIFF, ICO, HDR, EXR, TGA, PNM, QOI, DDS, Farbfeld to PNG.
- `render_image` tool views in-memory base64 or scratchpad bytes as a multimodal Image block.

### Changed

- Skills are now directory-based (`~/.config/agsh/skills/<name>/SKILL.md`), not flat files.
- System prompt lists skills by description and when_to_use; agent invokes via `skill` tool.
- `find_files` and `search_contents` descriptions recommend narrow searches, broadening gradually.
- Tool output redirected to scratchpad is never truncated; internal caps are lifted.
- Highlight markdown with `syntect` directly instead of `bat`; reprints are roughly 50x faster.
- Embed Monokai Extended theme from bat for visual parity with the old renderer.
- Drop the `Last message:` banner on session resume; the resuming-session line is sufficient.

### Fixed

- macOS/Windows CI tests no longer read the host user's real `config.toml` — they now isolate via `AGSH_CONFIG_DIR`.
- `cargo doc -D warnings` cleared of broken intra-doc links and bare-URL lints.
- Rename `render_image` input `scratchpad` to `from_scratchpad` so it no longer clobbers the source.
- Remove redundant 30 KB caps on `execute_command` and `spawn_agent`; oversize handled upstream.
- Show primary param in the tool banner for `skill` and `render_image`.

### Security

- Omit environment info (PWD, date, shell, OS) from prompts in `none` permission mode.

## [0.10.3] - 2026-04-14

### Fixed

- Fix newlines in tool/ask banners breaking single-line display.

## [0.10.2] - 2026-04-14

### Added

- CI workflow for `cargo fmt --check`, `cargo clippy`, and `cargo test`.
- Tests for `validate_tool_use_chains` in session resume.
- `SessionLockGuard` for panic-safe session unlocking.

### Changed

- Replace `let _ =` silent error discards with explicit handling.
- Extract CSS selectors to `LazyLock` statics in web search parsing.
- Deduplicate tool registration via shared `register_core_tools` helper.
- Replace busy-wait polling with blocking `recv()` in REPL event loop.
- Flatten `execute_tool_calls` into smaller helper methods.
- Resolve all clippy warnings (collapsible ifs, ptr_arg, etc.).

### Fixed

- Add `// SAFETY:` comment to `libc::kill` in session locking.

## [0.10.1] - 2026-04-14

### Fixed

- Fix code blocks rendered without newlines in bat mode.
- Fix extra blank lines after trailing code blocks.
- Fix blank lines between code blocks and surrounding content.

## [0.10.0] - 2026-04-14

### Added

- `/export` slash command to export the current session as Markdown.
- Re-print last message when resuming a session with `-c`.
- Adaptive thinking for Claude 4.6+ models.
- `set_thinking_override` on Provider trait for compaction.
- Optional `reasoning_effort` config for OpenAI o-series models.

### Changed

- Combine `-s` and `-c` CLI flags into `-c [SESSION_ID]`.
- Wrap injected context in `<context>` XML tags for structured parsing.
- Thinking enabled by default (was disabled).
- Default thinking budget: 10K → 16K tokens.
- Default max_tokens: 8K → 32K (non-thinking), dynamic (thinking).
- Preserve thinking blocks in conversation history for Claude API.
- Disable thinking during session compaction.
- Updated context window defaults for GPT-4.1 (1M) and o-series (200K).

### Fixed

- Session list preview now shows actual user input instead of "[Environment context]".

## [0.9.4] - 2026-04-14

### Added

- Output spacing state machine replacing ad-hoc separator flags.
- Blank line after tables in buffer via `normalize_spacing`.
- Validation of tool_use/tool_result chains on session resume.
- Warnings for unparseable messages during session loading.

### Fixed

- Fix missing blank line between tool batches and following text.
- Fix double blank line after todo list before text responses.
- Fix table not followed by blank line in bat render mode.
- Fix `normalize_spacing` splitting tables on incomplete streaming rows.
- Orphaned tool_use blocks no longer cause API errors on resume.

## [0.9.3] - 2026-04-13

### Added

- Table pretty-printing (column alignment) in bat render mode.

### Fixed

- Fix table column misalignment with emoji/wide Unicode characters.

## [0.9.2] - 2026-04-13

### Fixed

- Restore blank line after todo list to separate it from following tool calls.

## [0.9.1] - 2026-04-13

### Fixed

- Remove blank lines between consecutive tool call batches.
- Fix double blank line after todo list display.
- Blank line before text only prints when transitioning from tools.

## [0.9.0] - 2026-04-13

### Added

- `bat` render mode as the new default with syntax-highlighted markdown.

### Changed

- Rename `rich` render mode to `termimad` (`rich` kept as alias).
- Ensure blank line after markdown headers in bat/raw modes.
- Ensure proper spacing around tool indicator batches.

## [0.8.1] - 2026-04-13

### Changed

- Compaction now uses a structured summary prompt with 6 sections.
- Compaction preserves scratchpad entries and recent messages.
- Compaction re-injects environment, todos, and scratchpad inventory.
- Images and large text blocks stripped before summarization.

## [0.8.0] - 2026-04-13

### Added

- `replace_all` parameter for `edit_file` tool to replace all occurrences in a file.
- `force` parameter for `edit_file` tool to bypass the read-before-edit requirement.
- Read-before-edit enforcement: `edit_file` requires `read_file` on the same path first.
- `todo_write` tool for structured task tracking within a session.
- Ask permission mode (`a`): prompts user to approve/deny each tool call individually.
- Extended thinking support for the Claude provider (`[thinking]` config section).
- Image multimodal support: `read_file` returns base64-encoded images for `.png`/`.jpg`/`.gif`/`.webp`/`.bmp`.
- `TokenUsage` tracking parsed from Claude and OpenAI API responses.
- Auto-compact: automatically compacts conversation when input tokens exceed 80% of context window.
- `spawn_agent` tool for delegating research tasks to read-only sub-agents.
- Deferred tool loading: MCP tools listed in system prompt but schemas sent on first use.
- `raw` parameter for `fetch_url` tool to return untreated HTML instead of markdown.
- Scratchpad: session-scoped, name-keyed agent working memory.
- `scratchpad_write`, `scratchpad_read`, `scratchpad_edit`, `scratchpad_list`, `scratchpad_delete` tools.
- `scratchpad` parameter on all tools to save output directly.
- Auto-persist for oversized tool results (>30K chars) with `{tool}_{N}` naming.
- Per-tool output caps to prevent context overflow.
- `read_file` defaults to 2000 lines and rejects images over 3.75MB.
- Session export resolves persisted large outputs back to full content.

### Changed

- Permission levels expanded from 3 to 4: none, read, ask, write.
- `ToolResult.content` changed from `String` to `Vec<ToolResultContent>` for multimodal support.
- `Provider::complete()` now returns `TokenUsage` alongside the message and stop reason.
- `edit_file` success message now reports the number of replacements made.
- Tool outputs tied to session lifecycle: deleted with session/messages cleanup.

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
