# Claude subscription

The **Anthropic Messages API** billed to a Claude subscription. Authenticates by OAuth and mimics the Claude Code CLI's exact request shape, headers and request signing. Use this instead of a per-token Anthropic API key; for that, see [`anthropic-messages`](./anthropic-messages.md), which speaks the same protocol.

Named for the subscription rather than the protocol because that is what you are choosing: the endpoint is always `api.anthropic.com` and the client shape comes with the billing relationship.

> **Note:** This backend replicates Claude Code's fingerprinting and attestation machinery exactly. Modifying the request body, headers, or OAuth flow will cause requests to be rejected by Anthropic. If you hit 401/403 errors, verify that no middleware is rewriting the request.

## Configuration

| Setting | Value |
|---------|-------|
| Account `backend` | `claude-subscription` |
| Default base URL | `https://api.anthropic.com` |
| Credential | OAuth bundle kept in the store (acquired via `meka account add` / `login`) |
| Auth method | `Authorization: Bearer <oauth_token>` |
| API version | `2023-06-01` |

### Quickest start

```bash
meka account add anthropic --backend claude-subscription
meka profile add work --account anthropic --model claude-opus-5
```

`meka account add` prints an authorization URL for you to open, walks you through authorization,
and saves the tokens to the store under the `[accounts.anthropic]` table it writes.
`meka profile add` then names the model; a sole profile is the default.

### Config file

The two commands write this for you; you can also edit it by hand (secrets stay in the store):

```toml
default_profile = "work"

[accounts.anthropic]
backend = "claude-subscription"
# device_id, oauth_token_url, client_id are all optional overrides

[profiles.work]
account = "anthropic"
model = "claude-opus-5"
effort = "xhigh"         # optional; unset sends "high", as Claude Code does
thinking = "adaptive"    # optional; "adaptive"|"budgeted"|"off", default "adaptive"
thinking_display = "updates"  # optional; updates|summarized|redacted, default updates
```

See [Configuration → Config file](../configuration/config-file.md) for the full list of fields.

## Backend-specific keys

### `effort`

Sent as `output_config.effort` under the `effort-2025-11-24` beta. When unset, meka sends `high`, which is what Claude Code does; only a model that takes no effort at all gets neither the field nor the beta. An explicit value is absolute: sent verbatim, with no validation or clamping, whatever model it is aimed at. Typical values: `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`. See [Reasoning effort](#reasoning-effort).

### `thinking`

`adaptive` (the default) sends `thinking: {"type": "adaptive"}`; `budgeted` sends `{"type": "enabled", "budget_tokens": N}` from the profile's `thinking_budget` (falling back to [`[thinking].budget`](../configuration/config-file.md#thinkingbudget)), which pre-4.6 models require; `off` sends no thinking field. `temperature` follows whether thinking is on at all, not which encoding it uses. The betas do not: they are gated on the model alone.

### `thinking_display`

How the model's thinking is presented, one of Claude Code's three display modes. `updates`, the
default and Claude Code's own since 2.1.263, sends `thinking.display = "updates"` under the
`thinking-display-updates-2026-08-18` beta: the server streams a running token count instead of
the text, which the REPL draws as `Thinking... (150 tokens)`. `summarized` sends
`thinking.display = "summarized"` and streams a short readable summary. `redacted` sends the
`redact-thinking-2026-02-12` beta and no display field, and the server may answer with opaque
`redacted_thinking` blocks. In every mode the `thinking` blocks come back signed, and meka stores
and replays them verbatim, so multi-turn reasoning continuity is maintained. With thinking off
nothing is displayed and the redaction beta is sent, as Claude Code does.

A stored block records that its signature is Claude's, so resuming the session under an OpenAI profile does not replay a Claude signature as encrypted reasoning. A session recorded by 0.41 holds its blocks under a shape that names no provider, and meka does not reshape them when it opens a session; the [one-shot upgrade script](../getting-started/upgrading.md) does. Until it runs, such a block keeps its readable text and loses its signature, so those turns are not replayed as verified reasoning.

### `device_id`

Stable per-machine identifier embedded in `metadata.user_id` to mirror Claude Code's `~/.claude.json` device id (`getOrCreateUserID` in `utils/config.ts`).

If unset, meka first tries to adopt `userID` from `~/.claude.json` (so meka and Claude Code on the same machine present as the same device). If that file is missing or has no `userID`, meka generates a 64-character hex string. Either way the resolved value is persisted back to `[accounts.<name>].device_id` in `config.toml`. Other backends ignore this field; no stub config file is written for them.

### `client_id`

Optional override for the OAuth client id. Defaults to Claude Code's client id; rarely needed.

## Authentication

### OAuth login

`meka account add` (and `meka account login <name>` to re-authenticate) performs an OAuth 2.0 Authorization Code flow with PKCE:

1. meka generates a PKCE challenge and prints the URL of Claude's authorization page for you to open.
2. You authorize the application in your browser.
3. You paste the authorization code back into meka (the redirect URI is the platform.claude.com hosted callback page, not a local listener).
4. meka exchanges the code for access + refresh tokens.
5. Tokens are kept in the store and refreshed automatically.

The OAuth client id defaults to Claude Code's client id but can be overridden per account via `client_id`.

### Token lifecycle

1. Acquire the initial token with `meka account add` / `login`.
2. The token bundle is kept in the store, keyed by the account name.
3. On subsequent launches the token is loaded from the store.
4. meka refreshes the access token automatically when it's within 5 minutes of expiry; the new token is written back to the store under the same account.
5. If the refresh token dies, run `meka account login <name>` to re-authenticate. meka says so itself: a refresh the authorization server *rejects* ends the turn with that command in the error, naming the account. A refresh that fails because the token endpoint is rate-limited or down is retried with backoff instead, since neither answer means the grant is bad.

**Token refresh URL:** defaults to `https://platform.claude.com/v1/oauth/token`. Configurable via `oauth_token_url` on the account.

## Supported models

Any model your Claude Code subscription exposes. For the current line-up and their retirement dates, see [Anthropic's models overview](https://docs.claude.com/en/docs/about-claude/models/overview); `meka profile add` suggests `claude-opus-5` for a profile on a Claude account.

meka forwards the model string verbatim and doesn't gate which strings are valid. What is model-derived is a small set of gates, each pointed the way Claude Code points it. `temperature` is an allowlist, so an unrecognized model omits the field rather than earning a 400: it goes only to the models that still accept sampling params (Opus 4.6, Sonnet 4.6, Haiku 4.5, and older). `mid-conversation-system-2026-04-07` and `output_config.effort` are denylists, so an unrecognized model gets both: withholding the first would silently drop mid-conversation system messages, and effort is what a newer model is for. The `claude-code-20250219` beta is skipped for the Haiku tier. See [Beta header](#beta-header) and [Reasoning effort](#reasoning-effort).

## API details

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

Composed dynamically from the model, window and thinking settings, mirroring Claude Code's own assembly. Order is significant; the list below matches the Claude Code 2.1.263 interactive-CLI wire capture (tools present, thinking on, display updates) exactly:

| Beta | When |
|------|------|
| `claude-code-20250219` | All models *except* Haiku family |
| `oauth-2025-04-20` | Always (subscription auth) |
| `context-1m-2025-08-07` | The profile's `context_window` is a million tokens or more; Claude Code sends it for the `[1m]` model variant its user selected |
| `interleaved-thinking-2025-05-14` | Any modern Claude (4.x+) |
| `redact-thinking-2026-02-12` | Any modern Claude under `thinking_display = "redacted"`, or with thinking off unless the display is `summarized` |
| `thinking-token-count-2026-05-13` | Any modern Claude (4.x+) |
| `context-management-2025-06-27` | Any modern Claude (4.x+) |
| `prompt-caching-scope-2026-01-05` | Always |
| `mid-conversation-system-2026-04-07` | Everything except Claude 3.x, Opus 4.7 and older, Sonnet 4.6 and older, and Haiku 4.5 |
| `advanced-tool-use-2025-11-20` | When the request carries tools (meka always does) |
| `effort-2025-11-24` | Every model that takes an effort at all, whether or not the profile set one |
| `fallback-credit-2026-06-01` | Always. Claude Code latches it on every interactive turn; it only advertises that the server may answer with a fallback credit, and meka sends no `fallbacks` of its own |
| `thinking-display-updates-2026-08-18` | Any modern Claude with thinking on under `thinking_display = "updates"`, paired with `thinking.display = "updates"` |
| `extended-cache-ttl-2025-04-11` | Always (meka sends a 1h cache TTL) |
| `cache-diagnosis-2026-04-07` | Always, paired with the body's `diagnostics.previous_message_id`: the id of the previous response's message, or `null` on a conversation's first request and after a resume |

### System prompt

Sent as an array of three `text` blocks:

1. `x-anthropic-billing-header: cc_version=<version>.<fingerprint>; cc_entrypoint=cli; cch=<xxHash64-attestation>;` plus, when they apply, ` cc_is_subagent=true;`, ` cc_prev_req=<request id>;` and ` cc_prompt_id=<uuid>;`, in that order. The fingerprint suffix is a 3-character hex hash derived from the first user message (`SHA256(salt + msg[4] + msg[7] + msg[20] + version)[:3]`); the `cch` token is xxHash64 of a filtered copy of the serialized request body, computed and patched in just before send.

   `cc_prompt_id` identifies one human prompt and stays the same across every request that prompt produces, including the whole tool loop; a sub-agent inherits its spawner's. `cc_prev_req` names the `request-id` of the previous response in the same conversation, so it is absent on a conversation's first request. Both are absent from meka's own side queries, which is where Claude Code omits them too.
2. `You are Claude Code, Anthropic's official CLI for Claude.` (fixed identity prefix).
3. Your own system prompt, which carries `cache_control: {type: "ephemeral", ttl: "1h", scope: "global"}`.

Only block 3 is marked for caching, matching the captured Claude Code CLI wire; `scope: "global"` shares the cached prefix across sessions. Tools carry no `cache_control` (the rolling last-message breakpoint caches the tools+system prefix).

### Body key order

Keys are serialized in Claude Code's own order, which HTTP preserves:

```
model, messages, system, tools, metadata, max_tokens, thinking,
[temperature], [context_management], [output_config], [diagnostics], stream
```

Nothing in meka depends on that order. `patch_request_body` finds the `cch=00000` placeholder by walking the JSON structurally to the *top-level* `system` key rather than by searching for the billing header, so a conversation that quotes one (which any session about this code does) cannot capture the attestation.

### Other body fields

- `metadata.user_id`: JSON-encoded `{"device_id": "...", "account_uuid": "...", "session_id": "..."}` (`device_id` from the account's `device_id`; `account_uuid` from the OAuth token, empty until one is known; `session_id` is per-process).
- `context_management.edits = [{type: "clear_thinking_20251015", keep: "all"}]`: present when thinking is enabled on a context-management-capable model. Mirrors Claude Code's `apiMicrocompact`.
- `output_config.effort`: see [Reasoning effort](#reasoning-effort).
- `thinking.display`: see [`thinking_display`](#thinking_display); absent with thinking off and under `redacted`.
- `diagnostics.previous_message_id`: the id of the previous response's message in this conversation, `null` on the first request and after a resume; absent on a compaction request, which is a side query. Pairs with the `cache-diagnosis-2026-04-07` beta.
- `temperature: 1` (only when `thinking = "off"`, and only for models that still accept sampling params).
- `max_tokens`: `64_000` under `thinking = "adaptive"`, `max(thinking_budget * 2, 32_000)` under `budgeted`, `32_000` under `off`.

### Reasoning effort

Claude Code never leaves `output_config.effort` to the server on a model that takes one: it looks the model up in a table bundled in its binary, reads that model's `default_effort`, clamps it to what the model supports, and sends the result. meka also always sends a value, but one value rather than a per-model one, and sends the `effort-2025-11-24` beta alongside it.

| | sent |
|---|---|
| profile sets `effort` | that value, verbatim |
| profile sets nothing | `high` |
| model takes no effort | nothing, and no beta; a configured value is dropped with a warning |

One value for every model, not a copy of that table. `high` is what Claude Code's own resolution produces for almost every effort-capable model in the 2.1.263 table once the clamps have run, and it is what Claude Code falls back to for any model the table does not list. Carrying the per-model figures instead would add facts about Anthropic's data that go stale on their release schedule and buy nothing, because the server cannot tell a default meka chose from a value you configured. Models that take no effort at all are the Claude 3.x line, Opus 4.0/4.1, Sonnet 4.0/4.5 and Haiku 4.5.

A value you configure is absolute. Claude Code silently lowers `xhigh` or `max` to `high` on a model whose bundled entry lacks the capability; meka does not, because that table is a snapshot of someone else's system and quietly overriding what you asked for on the strength of it is worse than letting the API answer.

Only `claude-subscription` does this. `anthropic-messages` still omits `effort` when the profile sets none, because it can point at any Anthropic-compatible endpoint and has no standing to assert a default there.

### Cache control

The most recent message's last content block and the user system prompt carry `cache_control: {type: "ephemeral", ttl: "1h"}`. The 1h TTL is what an OAuth subscriber's Claude Code turn carries on the wire.

Caching is prefix-based: the system prompt precedes the tools array, which precedes the messages, so a byte changing early invalidates everything after it. meka is built so that nothing which changes mid-session sits in that prefix.

- **The system prompt is fixed for a session.** It carries only the role description, permission model, standing instructions, guidelines, and OS/shell info, all resolved once at startup. The tool catalog, skill list, and MCP server instructions live in the per-turn `<context>` block instead, because all three can change while a session runs.
- **The tools array only grows at the tail.** `load_tool` appends a schema rather than reordering, so the earlier entries stay byte-identical.
- **Permission toggles cost nothing.** See [Permissions](../usage/permissions.md).

Two things do legitimately invalidate it, both by necessity rather than oversight: compaction, which rewrites the head of the conversation, and an MCP server withdrawing a tool via `tools/list_changed`, which has to be removed from the tools array. The latter is confined to the array, leaving the system prompt ahead of it intact.

You can see the effect directly: `/status` reports the cache hit ratio, and reads should dominate from the second turn onward.

### Streaming

Server-Sent Events with the same event taxonomy as [`anthropic-messages`](./anthropic-messages.md): `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`. Reasoning streams as `thinking_delta` events; redacted thinking arrives as a `redacted_thinking` block carrying an opaque `data` payload and no signature, rendered as `[redacted thinking]`.
