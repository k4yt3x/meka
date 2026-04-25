# Claude API Provider

The `claude-api` provider talks to the [Claude Messages API](https://docs.anthropic.com/en/api/messages) directly with an `x-api-key` header. Use this when you have a Claude API key (billed per-token). For Claude Code OAuth, see [`claude-oauth`](./claude-oauth.md).

## Configuration

| Setting | Value |
|---------|-------|
| Provider name | `claude-api` |
| Default base URL | `https://api.anthropic.com` |
| API key env var | `CLAUDE_API_KEY` |
| Auth method | `x-api-key` header |
| API version | `2023-06-01` |

### Minimal Setup

```bash
export AGSH_PROVIDER=claude-api
export AGSH_MODEL=claude-opus-4-6
export CLAUDE_API_KEY=sk-ant-api03-...
agsh
```

### Config File

```toml
[provider]
name = "claude-api"
model = "claude-opus-4-6"
```

## Supported Models

Any model available through the Claude Messages API. Current line-up (per [Anthropic's models overview](https://docs.claude.com/en/docs/about-claude/models/overview)):

| Family | Alias | Notes |
|--------|-------|-------|
| Opus 4.7 | `claude-opus-4-7` | Latest Opus — most capable, adaptive thinking |
| Sonnet 4.6 | `claude-sonnet-4-6` | Latest Sonnet — speed + intelligence balance |
| Haiku 4.5 | `claude-haiku-4-5` | Latest Haiku — fastest |

Older but still available: `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-opus-4-1`. Deprecated and retiring 2026-06-15: `claude-opus-4-20250514`, `claude-sonnet-4-20250514`.

## Custom Base URL

To use a Claude-API-compatible proxy or gateway:

```bash
agsh --provider claude-api --model claude-opus-4-6 --base-url https://my-proxy.example.com
```

## API Details

**Endpoint:** `POST {base_url}/v1/messages`

**Headers:**
- `x-api-key: <api_key>`
- `anthropic-version: 2023-06-01`
- `content-type: application/json`

**System prompt:** Sent as a top-level `system` string.

**Tool format:** Tools are defined with `input_schema`:

```json
{
  "name": "read_file",
  "description": "Read the contents of a file at the given path.",
  "input_schema": { "type": "object", "properties": { ... } }
}
```

**Streaming:** Server-Sent Events with named event types (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, `ping`).
