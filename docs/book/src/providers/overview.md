# Providers Overview

Providers are the LLM inference backends that agsh uses to process your instructions. agsh ships with four built-in providers:

| Provider | Auth | API | Notes |
|----------|------|-----|-------|
| [`openai-api`](./openai-api.md) | `OPENAI_API_KEY` | Chat Completions | Works with OpenAI and any compatible endpoint (Ollama, vLLM, OpenRouter, …) |
| [`openai-codex`](./openai-codex.md) | OAuth login (setup wizard) | OpenAI Responses | Uses a ChatGPT subscription; talks to `chatgpt.com/backend-api/codex` like the Codex CLI |
| [`claude-api`](./claude-api.md) | `CLAUDE_API_KEY` | Claude Messages | Direct Claude API, billed per-token |
| [`claude-oauth`](./claude-oauth.md) | OAuth login (setup wizard) | Claude Messages | Uses a Claude Code subscription; replicates Claude Code's request shape and attestation |

## Selecting a Provider

Set the provider via any configuration layer:

```bash
# CLI flag
agsh --provider claude-oauth

# Environment variable
export AGSH_PROVIDER=claude-api

# Config file (~/.config/agsh/config.toml)
[provider]
name = "openai-api"
```

## OpenAI-Compatible APIs

The `openai-api` provider works with any API that implements the OpenAI Chat Completions format. This includes:

- **OpenAI** (default endpoint)
- **Ollama** (`http://localhost:11434/v1`)
- **OpenRouter** (`https://openrouter.ai/api/v1`)
- **vLLM**, **LiteLLM**, and other OpenAI-compatible servers

Set the `--base-url` flag or `OPENAI_BASE_URL` environment variable to point at the alternative endpoint.

## claude-api vs claude-oauth

Both talk to Claude's `/v1/messages` endpoint, but the auth and request shape differ:

- **`claude-api`** is the straightforward path — an `x-api-key` header, a plain system prompt, no extra headers. Choose this when you have a Claude API key.
- **`claude-oauth`** replicates the Claude Code CLI exactly: OAuth tokens, fingerprint-encoded version header, xxHash64 attestation over the request body, injected billing system block. Choose this when you want to use a Claude Code subscription. Any deviation from the expected shape causes requests to be rejected, so avoid proxies that rewrite headers or reformat the body.

## openai-api vs openai-codex

The two OpenAI-flavoured providers hit different endpoints with different protocols:

- **`openai-api`** posts to `/chat/completions` on `api.openai.com` (or any compatible endpoint), authenticating with an API key. This is the right choice when you have an OpenAI billing account or are pointing at a self-hosted OpenAI-compatible server.
- **`openai-codex`** posts to `chatgpt.com/backend-api/codex/responses` using the **OpenAI Responses API** (a different protocol — different request body shape, different streaming events). Authentication is OAuth against `auth.openai.com`, mirroring the first-party Codex CLI. Choose this to use a ChatGPT Plus / Pro / Team / Business subscription instead of a per-token API key.

## Streaming vs Non-Streaming

By default, agsh uses streaming mode: tokens appear in the terminal as they are generated. Use `--no-stream` to wait for the complete response before displaying it.

Streaming is recommended for interactive use. Non-streaming may be useful for scripting or when the provider does not support SSE.
