# Environment Variables

The [config file](./config-file.md) is the recommended way to configure agsh. Environment variables are useful as overrides -- for example, in CI pipelines, containers, or when you want to temporarily switch providers without editing your config.

Environment variables override config file values but are overridden by CLI flags.

## agsh-Specific Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AGSH_PROVIDER` | LLM provider name | `openai-api`, `claude-api`, `claude-oauth` |
| `AGSH_MODEL` | Model identifier | `gpt-4o`, `claude-sonnet-4-20250514` |
| `AGSH_PERMISSION` | Default permission mode | `none`, `read`, `write` |
| `AGSH_CONFIG_DIR` | Override the default config directory. Points at the `agsh` directory itself (contains `config.toml` and `skills/`). The only isolation knob that works on every platform — `dirs::config_dir()` ignores `$XDG_CONFIG_HOME` on macOS/Windows. | `/tmp/agsh-test/agsh` |

## MCP Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AGSH_MCP_TOOL_TIMEOUT` | Per-call timeout for MCP tools, in milliseconds. Applies to every remote tool invocation; on timeout agsh cancels the request and returns an error to the model. | `600000` (600s) |

## Provider API Keys

| Variable | Used When |
|----------|-----------|
| `OPENAI_API_KEY` | Provider is `openai-api` |
| `CLAUDE_API_KEY` | Provider is `claude-api` |

## OAuth Authentication

| Variable | Description |
|----------|-------------|
| `CLAUDE_OAUTH_TOKEN` | OAuth access token for the `claude-oauth` provider |
| `OPENAI_CODEX_TOKEN` | OAuth access token for the `openai-codex` provider |
| `CODEX_CLIENT_ID` | Override the default OpenAI OAuth client ID for the `openai-codex` setup wizard (rarely needed) |

On first use, the OAuth token is saved to the database and loaded automatically on subsequent launches. Setting the env var again replaces the stored token.

## Provider Base URL

| Variable | Description |
|----------|-------------|
| `OPENAI_BASE_URL` | Custom base URL for the OpenAI-compatible endpoint |

## Logging

agsh uses the `tracing` framework. The log level can be controlled with:

| Variable | Description | Example |
|----------|-------------|---------|
| `RUST_LOG` | Standard Rust log filter | `agsh=debug`, `agsh=trace` |

If `RUST_LOG` is not set, the verbosity flag (`-v`, `-vv`, `-vvv`) controls the level:

| Flag | Level |
|------|-------|
| (none) | `warn` |
| `-v` | `info` |
| `-vv` | `debug` |
| `-vvv` | `trace` |

Logs are written to stderr so they do not interfere with agent output.
