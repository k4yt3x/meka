# Providers Overview

Providers are the LLM inference backends that agsh uses to process your instructions. agsh ships with two built-in providers:

| Provider | API | Streaming | Tool Calling |
|----------|-----|-----------|-------------|
| [OpenAI](./openai.md) | Chat Completions | SSE | Function calling |
| [Claude](./claude.md) | Messages API | SSE (named events) | Content blocks |

## Selecting a Provider

Set the provider via any configuration layer:

```bash
# CLI flag
agsh --provider openai

# Environment variable
export AGSH_PROVIDER=claude

# Config file (~/.config/agsh/config.toml)
[provider]
name = "openai"
```

## OpenAI-Compatible APIs

The `openai` provider works with any API that implements the OpenAI Chat Completions format. This includes:

- **OpenAI** (default endpoint)
- **Ollama** (`http://localhost:11434/v1`)
- **OpenRouter** (`https://openrouter.ai/api/v1`)
- **vLLM**, **LiteLLM**, and other OpenAI-compatible servers

Set the `--base-url` flag or `OPENAI_BASE_URL` environment variable to point to the alternative endpoint.

## Streaming vs Non-Streaming

By default, agsh uses streaming mode: tokens appear in the terminal as they are generated. Use `--no-stream` to wait for the complete response before displaying it.

Streaming is recommended for interactive use. Non-streaming may be useful for scripting or when the provider does not support SSE.
