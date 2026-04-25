# Claude OAuth Provider

The `claude-oauth` provider authenticates with a Claude Code subscription via OAuth and mimics the Claude Code CLI's exact request shape, headers, and request signing. Use this when you have a Claude Code subscription instead of a per-token Claude API key. For the direct Messages API, see [`claude-api`](./claude-api.md).

> **Note:** This provider replicates Claude Code's fingerprinting and attestation machinery exactly. Modifying the request body, headers, or OAuth flow will cause requests to be rejected by Anthropic. If you hit 401/403 errors, verify that no middleware is rewriting the request.

## Configuration

| Setting | Value |
|---------|-------|
| Provider name | `claude-oauth` |
| Default base URL | `https://api.anthropic.com` |
| OAuth token env var | `CLAUDE_OAUTH_TOKEN` |
| OAuth client ID env var | `CLAUDE_CLIENT_ID` (optional) |
| Auth method | `Authorization: Bearer <oauth_token>` |
| API version | `2023-06-01` |

### Quickest Start (Setup Wizard)

```bash
agsh setup
```

Pick **claude-oauth** when prompted. The wizard opens your browser, walks you through authorization, and saves the tokens to the local database.

### Minimal Setup (Manual OAuth Token)

```bash
export AGSH_PROVIDER=claude-oauth
export AGSH_MODEL=claude-opus-4-6
export CLAUDE_OAUTH_TOKEN=sk-ant-oat01-...
agsh
```

On first run the OAuth token is saved to the database. Subsequent runs load it automatically; you no longer need the env var.

### Config File

```toml
[provider]
name = "claude-oauth"
model = "claude-opus-4-6"
effort = "high"          # optional; "low" | "medium" | "high"
redact_thinking = false  # optional; redact thinking content for privacy
# device_id, oauth_token_url, oauth_token are all optional overrides
```

See [Configuration → Config File](../configuration/config-file.md) for the full list of fields.

## Provider-specific knobs

### `[provider].effort`

Sent as `output_config.effort` for effort-capable models (`opus-4-6`, `sonnet-4-6`). Accepts `"low"`, `"medium"`, or `"high"`. Defaults to `"high"`. Mirrors Claude Code's effort knob in `utils/effort.ts`. Older models (Sonnet 4.0, Opus 4.1, Haiku 4.5) ignore this field on the wire and the body field is omitted automatically.

### `[provider].redact_thinking`

When `true`, agsh adds the `redact-thinking-2026-02-12` beta header so the server returns redacted thinking blocks instead of full thinking summaries. The redacted payloads can't be replayed back to the server in multi-turn conversations, so agsh stores them as opaque signatures only. Defaults to `false`.

### `[provider].device_id`

Stable per-machine identifier embedded in `metadata.user_id` to mirror Claude Code's `~/.claude.json` device ID (`getOrCreateUserID` in `utils/config.ts`).

If unset, agsh first tries to adopt `userID` from `~/.claude.json` (so agsh and Claude Code on the same machine present as the same device). If that file is missing or has no `userID`, agsh generates a 64-character hex string. Either way the resolved value is persisted back to `[provider].device_id` in `config.toml`. Other providers ignore this field — no stub config file is written for them.

## Authentication

### Setup Wizard (recommended)

`agsh setup` performs an OAuth 2.0 Authorization Code flow with PKCE:

1. agsh generates a PKCE challenge and opens your browser to Claude's authorization page.
2. You authorize the application in your browser.
3. You paste the authorization code back into agsh (the redirect URI is the platform.claude.com hosted callback page, not a local listener).
4. agsh exchanges the code for access + refresh tokens.
5. Tokens are stored in the local database and refreshed automatically.

The OAuth client ID defaults to Claude Code's client ID but can be overridden via the `CLAUDE_CLIENT_ID` env var.

### Token Lifecycle

1. Provide the initial token via setup wizard, env var, or config.
2. agsh saves it to the database on first use.
3. On subsequent launches the token is loaded from the database.
4. agsh refreshes the access token automatically when it's within 5 minutes of expiry; the new token is written back to the database.
5. Setting a new env var or config value replaces the stored token.

**Token refresh URL:** defaults to `https://api.anthropic.com/v1/oauth/token`. Configurable via `provider.oauth_token_url` in the config file.

## Supported Models

Any model your Claude Code subscription exposes. Current line-up (per [Anthropic's models overview](https://docs.claude.com/en/docs/about-claude/models/overview)):

| Family | Alias | Notes |
|--------|-------|-------|
| Opus 4.7 | `claude-opus-4-7` | Latest Opus — most capable, no extended-thinking, adaptive thinking |
| Sonnet 4.6 | `claude-sonnet-4-6` | Latest Sonnet — speed + intelligence balance |
| Haiku 4.5 | `claude-haiku-4-5` | Latest Haiku — fastest |

Older but still available: `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-opus-4-1`. Deprecated and retiring 2026-06-15: `claude-opus-4-20250514`, `claude-sonnet-4-20250514`.

agsh forwards the model string verbatim — it doesn't gate which strings are valid. Per-model behaviour depends on capability gates baked into the request shape (see [Beta header](#beta-header)). The current gates target `opus-4-6` / `sonnet-4-6` for adaptive-thinking and effort; newer models (e.g. Opus 4.7) fall through to the conservative defaults until the gates are updated.

## API Details

**Endpoint:** `POST {base_url}/v1/messages?beta=true`

**Authentication & identity headers:**

- `Authorization: Bearer <oauth_token>`
- `anthropic-version: 2023-06-01`
- `anthropic-beta: <comma-separated beta list>` (computed per request, see below)
- `x-app: cli`
- `User-Agent: claude-cli/<version> (external, cli)`
- `X-Claude-Code-Session-Id: <uuid>` (per-process)
- Stainless SDK identification headers (`x-stainless-*`)

### Beta header

Composed dynamically from the model + thinking settings, mirroring Claude Code's `getAllModelBetas` (`utils/betas.ts`). Order is significant — wire dumps from Claude Code show this exact ordering:

| Beta | When |
|------|------|
| `claude-code-20250219` | All models *except* Haiku family |
| `oauth-2025-04-20` | Always (subscription auth) |
| `adaptive-thinking-2026-01-28` | Thinking on AND model is `opus-4-6` / `sonnet-4-6` |
| `interleaved-thinking-2025-05-14` | Thinking on AND model is older Claude 4 (Sonnet 4.0, etc.) |
| `redact-thinking-2026-02-12` | `[provider].redact_thinking = true` AND thinking on |
| `context-management-2025-06-27` | Any modern Claude (4.x+) |
| `prompt-caching-scope-2026-01-05` | Always |
| `effort-2025-11-24` | `opus-4-6` / `sonnet-4-6` only |

### System prompt

Sent as an array of three `text` blocks:

1. `x-anthropic-billing-header: cc_version=<version>.<fingerprint>; cc_entrypoint=cli; cch=<xxHash64-attestation>;` — the fingerprint suffix is a 3-character hex hash derived from the first user message (`SHA256(salt + msg[4] + msg[7] + msg[20] + version)[:3]`); the `cch` token is xxHash64 of the entire serialized request body, computed and patched in just before send.
2. `You are Claude Code, Anthropic's official CLI for Claude.` — fixed identity prefix.
3. Your own system prompt — carries `cache_control: {type: "ephemeral", ttl: "1h"}`.

Only block 3 is marked for caching, matching the recent Claude Code wire shape ("boundary mode" in `utils/api.ts:362-409`). Blocks 1 and 2 must come first so the `cch=00000` placeholder is the first occurrence in the serialized JSON, which is what `patch_request_body` looks for when computing the attestation.

### Other body fields

- `metadata.user_id`: JSON-encoded `{"device_id": "...", "account_uuid": "", "session_id": "..."}` — `device_id` from `[provider].device_id`, `session_id` is per-process.
- `context_management.edits = [{type: "clear_thinking_20251015", keep: "all"}]` — present when thinking is enabled on a context-management-capable model. Mirrors Claude Code's `apiMicrocompact`.
- `output_config.effort`: present for effort-capable models, value from `[provider].effort`.
- `temperature: 1` — only when thinking is disabled.
- `max_tokens` — `64_000` for adaptive-thinking models, `max(thinking_budget * 2, 32_000)` for legacy thinking models, `32_000` otherwise.

### Cache control

The most recent message's last content block, the last tool definition, and the user system prompt all carry `cache_control: {type: "ephemeral", ttl: "1h"}`. The 1h TTL matches Claude Code's `getCacheControl` for OAuth subscribers (`should1hCacheTTL` in `claude.ts:358-374`). Mid-session permission toggles never invalidate this cache — see [Permissions](../usage/permissions.md) for the reasoning.

### Streaming

Server-Sent Events with the same event taxonomy as [`claude-api`](./claude-api.md): `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`. Reasoning streams as `thinking_delta` events; redacted thinking arrives as a single `[redacted]` block plus a signature.
