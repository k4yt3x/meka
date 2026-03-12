# Claude Provider

The Claude provider uses the [Claude API Messages endpoint](https://docs.anthropic.com/en/api/messages).

## Configuration

| Setting | Value |
|---------|-------|
| Provider name | `claude` |
| Default base URL | `https://api.anthropic.com` |
| API key env var | `CLAUDE_API_KEY` |
| OAuth token env var | `CLAUDE_OAUTH_TOKEN` |
| Auth method | `x-api-key` header (API key) or `Authorization: Bearer` (OAuth) |
| API version | `2023-06-01` |
| Max tokens | `8192` |

### Quickest Start (OAuth Login)

Run the setup wizard and choose **OAuth login** when prompted:

```bash
agsh setup
```

This opens your browser for authorization, exchanges the code for tokens, and saves them to the database. No API key needed.

### Minimal Setup (API Key)

```bash
export AGSH_PROVIDER=claude
export AGSH_MODEL=claude-sonnet-4-20250514
export CLAUDE_API_KEY=sk-ant-api03-...
agsh
```

### Minimal Setup (OAuth Token)

```bash
export AGSH_PROVIDER=claude
export AGSH_MODEL=claude-sonnet-4-20250514
export CLAUDE_OAUTH_TOKEN=sk-ant-oat01-...
agsh
```

On the first run, the OAuth token is saved to the database. On subsequent runs, the token is loaded automatically without needing the environment variable.

### Config File

```toml
[provider]
name = "claude"
model = "claude-sonnet-4-20250514"
```

## Authentication

agsh supports two authentication methods for the Claude provider:

### OAuth Login

The recommended way to authenticate. Run `agsh setup` (or let the first-launch wizard guide you) and select **OAuth login**. This performs an OAuth Authorization Code flow with PKCE:

1. agsh generates a PKCE challenge and opens your browser to Claude's authorization page
2. You authorize the application in your browser
3. You paste the authorization code back into agsh
4. agsh exchanges the code for access and refresh tokens
5. Tokens are stored in the database and refreshed automatically

The OAuth client ID defaults to Claude Code's client ID but can be overridden via the `CLAUDE_CLIENT_ID` environment variable.

### API Key

Traditional API key authentication using the `x-api-key` header. Set via `CLAUDE_API_KEY` env var or `provider.api_key` in the config file.

### Manual OAuth Token

OAuth token authentication using the `Authorization: Bearer` header. Set via `CLAUDE_OAUTH_TOKEN` env var or `provider.oauth_token` in the config file.

OAuth tokens are automatically detected by their `sk-ant-oat01-` prefix, even when passed via `CLAUDE_API_KEY`.

**Token lifecycle:**

1. Provide the initial token via env var, config, or OAuth login
2. agsh saves it to the database on first use
3. On subsequent launches, the token is loaded from the database
4. If the token expires, agsh refreshes it automatically and updates the database
5. Setting a new env var or config value replaces the stored token

**Token refresh URL:** Defaults to `https://console.anthropic.com/v1/oauth/token`. Configurable via `provider.oauth_token_url` in the config file.

## Supported Models

Any model available through the Claude API:

- `claude-opus-4-20250514`
- `claude-sonnet-4-20250514`
- `claude-haiku-4-5-20251001`

## Custom Base URL

To use a Claude-compatible proxy or gateway:

```bash
agsh --provider claude --model claude-sonnet-4-20250514 --base-url https://my-proxy.example.com
```

## API Details

**Endpoint:** `POST {base_url}/v1/messages`

**Headers (API key):**
- `x-api-key: <api_key>`
- `anthropic-version: 2023-06-01`
- `content-type: application/json`

**Headers (OAuth):**
- `Authorization: Bearer <oauth_token>`
- `anthropic-version: 2023-06-01`
- `content-type: application/json`

**System prompt:** Sent as a top-level `system` field in the request body (not as a message).

**Tool format:** Tools are defined with `input_schema` instead of `parameters`:

```json
{
  "name": "read_file",
  "description": "Read the contents of a file at the given path.",
  "input_schema": { "type": "object", "properties": { ... } }
}
```

**Tool use and results:** Expressed as content blocks within messages:

- Tool use: `{"type": "tool_use", "id": "...", "name": "...", "input": {...}}`
- Tool result: `{"type": "tool_result", "tool_use_id": "...", "content": "..."}`

**Streaming:** Uses Server-Sent Events with named event types:

| Event | Description |
|-------|-------------|
| `message_start` | Message initialization |
| `content_block_start` | Begin a text or tool_use block |
| `content_block_delta` | Incremental text (`text_delta`) or tool input (`input_json_delta`) |
| `content_block_stop` | End of a content block |
| `message_delta` | Final metadata including `stop_reason` |
| `message_stop` | Stream complete |
| `ping` | Keep-alive |
