# Config File

agsh looks for a TOML configuration file at:

```
~/.config/agsh/config.toml
```

More precisely, it uses the XDG config directory (`$XDG_CONFIG_HOME/agsh/config.toml`), which defaults to `~/.config/agsh/config.toml` on Linux.

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
| `anthropic` | Anthropic Messages API |

### `provider.model`

The model identifier to send to the provider. Examples:

- `gpt-4o`, `gpt-4o-mini` (OpenAI)
- `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` (Anthropic)
- Any model supported by an OpenAI-compatible endpoint

### `provider.api_key`

The API key for authentication. It is recommended to use environment variables (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) instead of storing the key in the config file.

### `provider.base_url`

Custom API base URL. Useful for:

- Self-hosted models via [Ollama](https://ollama.ai) (`http://localhost:11434/v1`)
- [OpenRouter](https://openrouter.ai) (`https://openrouter.ai/api/v1`)
- Other OpenAI-compatible API providers

If not set, defaults to:

- `https://api.openai.com/v1` for the `openai` provider
- `https://api.anthropic.com` for the `anthropic` provider

## Examples

### OpenAI

```toml
[provider]
name = "openai"
model = "gpt-4o"
# API key via env: export OPENAI_API_KEY=sk-...
```

### Anthropic

```toml
[provider]
name = "anthropic"
model = "claude-sonnet-4-20250514"
# API key via env: export ANTHROPIC_API_KEY=sk-ant-...
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
