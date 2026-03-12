# Environment Variables

The [config file](./config-file.md) is the recommended way to configure agsh. Environment variables are useful as overrides -- for example, in CI pipelines, containers, or when you want to temporarily switch providers without editing your config.

Environment variables override config file values but are overridden by CLI flags.

## agsh-Specific Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AGSH_PROVIDER` | LLM provider name | `openai`, `claude` |
| `AGSH_MODEL` | Model identifier | `gpt-4o`, `claude-sonnet-4-20250514` |
| `AGSH_PERMISSION` | Default permission mode | `none`, `read`, `write` |

## Provider API Keys

| Variable | Used When |
|----------|-----------|
| `OPENAI_API_KEY` | Provider is `openai` |
| `CLAUDE_API_KEY` | Provider is `claude` |

## OAuth Authentication

| Variable | Description |
|----------|-------------|
| `CLAUDE_OAUTH_TOKEN` | OAuth access token for the Claude provider |

OAuth tokens (with `sk-ant-oat01-` prefix) are also auto-detected when passed via `CLAUDE_API_KEY`.

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
