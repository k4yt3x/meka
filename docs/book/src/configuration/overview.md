# Configuration Overview

The recommended way to configure agsh is with a config file at `~/.config/agsh/config.toml`:

```toml
[provider]
name = "openai-api"
model = "gpt-4o"
api_key = "sk-..."
```

This is all you need to get started. See [Config File](./config-file.md) for the full reference.

## Required Settings

agsh requires three settings to function. If any are missing, it prints an error with setup instructions:

| Setting | Config Key | Env Var | CLI Flag |
|---------|------------|---------|----------|
| Provider | `provider.name` | `AGSH_PROVIDER` | `--provider` |
| Model | `provider.model` | `AGSH_MODEL` | `-m`, `--model` |
| API Key | `provider.api_key` | `OPENAI_API_KEY` or `CLAUDE_API_KEY` | -- |

## Override Layers

Configuration is layered. Higher-priority layers override lower ones:

1. **CLI flags** -- per-invocation overrides (`--provider`, `--model`, `--base-url`, `-p`)
2. **Environment variables** -- useful for CI, containers, or temporary overrides (`AGSH_PROVIDER`, etc.)
3. **Config file** -- persistent settings in `~/.config/agsh/config.toml`
4. **Built-in defaults** -- permission defaults to `read`, streaming defaults to on

For example, `--model gpt-4o-mini` on the command line overrides both `AGSH_MODEL` and `provider.model` in the config file.

## API Key Resolution

The API key environment variable depends on the configured provider:

- Provider `openai-api`: reads `OPENAI_API_KEY`
- Provider `openai-codex`: reads `OPENAI_CODEX_TOKEN`, otherwise loads the OAuth token saved by `agsh setup`
- Provider `claude-api`: reads `CLAUDE_API_KEY`
- Provider `claude-oauth`: reads `CLAUDE_OAUTH_TOKEN`, otherwise loads the OAuth token saved by `agsh setup`

If the environment variable is not set, it falls back to `provider.api_key` (or `provider.oauth_token` for OAuth providers) in the config file.
