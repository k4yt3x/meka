# Config File

agsh looks for a TOML configuration file at a platform-specific location:

| Platform | Path |
|----------|------|
| Linux | `~/.config/agsh/config.toml` (`$XDG_CONFIG_HOME/agsh/config.toml`) |
| macOS | `~/Library/Application Support/agsh/config.toml` |
| Windows | `%APPDATA%\agsh\config.toml` |

The config file is optional. If it does not exist, agsh silently skips it.

## Format

```toml
[provider]
name = "openai"
model = "gpt-4o"
api_key = "sk-..."
base_url = "https://api.openai.com/v1"
```

All fields under `[provider]` are optional individually -- you can set some in the config file and override others with environment variables or CLI flags.

## Fields

### `provider.name`

The LLM provider to use.

| Value | Description |
|-------|-------------|
| `openai` | OpenAI Chat Completions API (also works with OpenAI-compatible APIs) |
| `claude` | Claude API (Messages endpoint) |

### `provider.model`

The model identifier to send to the provider. Examples:

- `gpt-4o`, `gpt-4o-mini` (OpenAI)
- `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` (Claude)
- Any model supported by an OpenAI-compatible endpoint

### `provider.api_key`

The API key for authentication. It is recommended to use environment variables (`OPENAI_API_KEY` or `CLAUDE_API_KEY`) instead of storing the key in the config file.

### `provider.oauth_token`

OAuth access token for the Claude provider. Can also be set via `CLAUDE_OAUTH_TOKEN` env var. The token is saved to the database on first use and loaded automatically on subsequent launches.

### `provider.oauth_token_url`

Custom OAuth token refresh endpoint. Defaults to `https://console.anthropic.com/v1/oauth/token`.

### `provider.base_url`

Custom API base URL. Useful for:

- Self-hosted models via [Ollama](https://ollama.ai) (`http://localhost:11434/v1`)
- [OpenRouter](https://openrouter.ai) (`https://openrouter.ai/api/v1`)
- Other OpenAI-compatible API providers

If not set, defaults to:

- `https://api.openai.com/v1` for the `openai` provider
- `https://api.anthropic.com` for the `claude` provider

## Examples

### OpenAI

```toml
[provider]
name = "openai"
model = "gpt-4o"
# API key via env: export OPENAI_API_KEY=sk-...
```

### Claude

```toml
[provider]
name = "claude"
model = "claude-sonnet-4-20250514"
# API key via env: export CLAUDE_API_KEY=sk-ant-api03-...
# Or OAuth token via env: export CLAUDE_OAUTH_TOKEN=sk-ant-oat01-...
```

### Ollama (local)

```toml
[provider]
name = "openai"
model = "llama3"
api_key = "unused"
base_url = "http://localhost:11434/v1"
```

### OpenRouter

```toml
[provider]
name = "openai"
model = "anthropic/claude-sonnet-4-20250514"
base_url = "https://openrouter.ai/api/v1"
# API key via env: export OPENAI_API_KEY=sk-or-...
```

## `[web]`

Settings for web-related tools (fetch_url, web_search).

### `web.user_agent`

Custom User-Agent string for HTTP requests. Some search engines may block requests with non-browser User-Agent strings.

Default: `Mozilla/5.0 (compatible; agsh/0.1)`

```toml
[web]
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
```

## `[shell]`

Settings for shell command execution.

### `shell.sandbox`

Whether to enable read-only filesystem sandboxing for shell commands in read mode. When enabled (default), shell commands can be executed in read mode but with the filesystem physically write-protected. When disabled, shell commands require write mode.

Default: `true`

```toml
[shell]
sandbox = false  # disable sandboxed shell in read mode
```

The sandbox uses Landlock on Linux (kernel 5.13+) and sandbox-exec on macOS. On platforms where sandboxing is unavailable, shell commands always require write mode regardless of this setting.

## `[session]`

Settings for session history retention and context window management.

### `session.context_messages`

Maximum number of messages to send to the LLM API per request. Older messages are truncated from the beginning while preserving tool call chain integrity. The full history remains stored in SQLite -- only the API payload is limited.

Default: `200`

```toml
[session]
context_messages = 100
```

### `session.retention_days`

Automatically delete sessions older than this many days on startup. Uses the session's `updated_at` timestamp, so actively-resumed sessions are preserved even if originally created long ago.

Default: `90`

```toml
[session]
retention_days = 30
```

### `session.max_storage_bytes`

Maximum total byte size of all stored message content across all sessions. When exceeded on startup, the oldest sessions are deleted until the total is under the limit.

Default: `52428800` (50 MB)

```toml
[session]
max_storage_bytes = 10485760  # 10 MB
```
