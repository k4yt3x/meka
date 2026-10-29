# MCP

The [Model Context Protocol](https://modelcontextprotocol.io) is how meka reaches tools, resources
and prompts it does not implement itself. A server is a process meka spawns or an HTTP endpoint it
connects to; what it advertises is registered alongside the built-in tools and called the same way.

This page covers running servers: the command suite, where their secrets live, what happens on the
wire, and what the agent can reach. The keys themselves are in the
[config file reference](../configuration/config-file.md#mcp), which is also where tool permissions
are resolved.

## `meka mcp` CLI

A running `meka serve` offers the same verbs over HTTP, on `/v1/mcp` under the `config:w` scope,
and a server added, changed or removed there joins, restarts on or leaves the running manager
without a restart; see
[Managing the installation](http-api.md#managing-the-installation).

Manage configured servers without editing `config.toml` by hand:

| Command | Action |
|---|---|
| `meka mcp list [--format json]` | Print all configured servers, plus any stored OAuth credential that no server claims (see [Leftover credentials](../configuration/config-file.md#leftover-credentials)). Under `json`, `{"servers": [...]}` with each server's `name`, `transport`, `required`, `disabled`, and its `permission`, `command`, `args` or `url` where set. |
| `meka mcp get <name> [--format json]` | Print full details for one server: the listing's fields, `deferral`, `inherited` (which of `required` and `deferral` the entry leaves to `[mcp]`), `env_keys`, `header_keys`, the kinds of `credentials` stored and presented to its `url` (never their values), `credentials_for_another_endpoint` (each kind stored for another endpoint, with the origin it was stored for), and every other key as written: `headers_helper`, the `auth` block, `allowed_tools`, `disabled_tools`, `eager_load_tools`, `trust_read_only_hint` and `tool_permissions`. |
| `meka mcp add <name> <url-or-command> [args...] [flags]` | Persist a server. Transport is auto-detected: a URL starting with `http[s]://` means HTTP, anything else means stdio. Preserves existing formatting/comments via `toml_edit`. |
| `meka mcp remove <name>` | Clear stored credentials, delete the server entry and drop any resource-update ledger entries, then revoke the stored OAuth tokens (RFC 7009) at the provider, best-effort. A name with stored credentials but no config entry is cleaned rather than refused. |
| `meka mcp disable <name>` | Set `disabled = true` on the server entry. The next `meka` start skips it entirely. |
| `meka mcp enable <name>` | Clear the `disabled` flag, so the server connects on the next start. |
| `meka mcp reconnect <name>` | Smoke-test a connect; exits non-zero with the error when it fails. |
| `meka mcp tools <name> [--format json]` | Connect and list every advertised tool with its resolved permission, the chain step that decided it, and whether the current config allows it. Useful for populating `--allow-tool`, `--disable-tool`, or `--tool-permission` overrides without leaving the CLI. Under `json`, the object `GET /v1/mcp/{name}/tools` answers with (`server`, `tools[]` of `raw_name`, `description`, `required_permission`, `permission_source`, `allowed`, `loading`) plus `read_only_hint_declined`. |
| `meka mcp login <name>` | Drive interactive OAuth. If the server has no `[auth]` block and uses HTTP, assumes `type = "oauth"` and persists the block on success. With `--auth-token-stdin` or `--client-secret-stdin`, stores that secret and exits instead, which is also how you rotate one. A pasted callback the flow cannot read, or one carrying another login's state, is refused and asked for again. |
| `meka mcp logout <name>` | Clear every stored credential for the server, then call the provider's `revocation_endpoint` (RFC 7009) for the OAuth tokens it held, best-effort. |

### Credentials

An MCP server's bearer token, OAuth client secret and OAuth token bundle are kept in the store (`mcp_credentials`, keyed by server name and kind, each bound to the endpoint it was stored for), never in `config.toml`. This is the same rule accounts follow, and for the same reason: `config.toml` is a plaintext file people commit, sync and share.

Each is read from stdin so it never reaches `ps` output or your shell history. One command reads one secret, so `--auth-token-stdin` and `--client-secret-stdin` cannot be combined:

```console
$ pass show notion-token | meka mcp add notion https://mcp.notion.com/mcp --auth-token-stdin
$ pass show acme-secret | meka mcp login acme --client-secret-stdin
```

A confidential OAuth client holds two at once: the long-lived client secret it authenticates with, and the refreshable bundle it obtained. Store the secret first, then run `meka mcp login <name>` to complete the flow. Refreshing the bundle leaves the client secret alone.

A credential is presented only where it was given: the OAuth tokens a login obtained to the exact `url` they were issued for, which is their audience, and a bearer or a client secret anywhere on the scheme, host and port of the `url` it was stored with. A `url` moved to another host takes nothing along, so a credential is never sent to a server it was not meant for. The server then fails to connect, saying what it holds and for where, until a login, or the secret stored again, gives it one for the new `url`; a bearer beside an `[auth]` block, or a client secret beside a `client_credentials_jwt` block, is never sent and never stops a connect. A credential is bound to the `url` as this process's environment expands it, so a `${VAR}` in the `url` that the environment leaves unset refuses storing a secret for it. `meka mcp get <name>` lists which kinds a server has, without printing any of them, and names under `not presented` each kind it holds for another endpoint with the origin it was stored for. `meka mcp list` names servers that have a stored credential but no `[[mcp.servers]]` entry, which is what a hand-edited config strands.

### `meka mcp add` flags

| Flag | Purpose |
|------|---------|
| `--transport <TRANSPORT>` | Force the transport, `stdio` or `http`; auto-detected otherwise. |
| `--env <KEY=VALUE>` | Environment variable for stdio (repeatable). |
| `--header <KEY=VALUE>` | HTTP header (repeatable). |
| `--auth <AUTH>` | Configure the `[auth]` block: `oauth`, `client_credentials` or `client_credentials_jwt`. |
| `--auth-token-stdin` | Read a static bearer token from stdin and store it. Mutually exclusive with `--auth`. |
| `--client-secret-stdin` | Read an OAuth client secret from stdin and store it. Required by `--auth client_credentials`. |
| `--client-id <CLIENT_ID>` | OAuth or client_credentials client id. Not a secret, so it goes in `config.toml`. |
| `--signing-key <SIGNING_KEY>`, `--signing-algorithm <SIGNING_ALGORITHM>` | JWT signing key path and algorithm (`RS256`, `RS384`, `RS512`, `ES256`, `ES384`), `client_credentials_jwt` only. |
| `--scope <SCOPE>` | OAuth scope (repeatable). |
| `--redirect-port <REDIRECT_PORT>` | Fixed OAuth redirect port (default: ephemeral). |
| `--permission <LEVEL>` | Per-server permission cap, applied to every tool on the server: `none`, `read`, `workspace` or `unrestricted`. Omitted, the level resolves as for any server without the key. |
| `--allow-tool <TOOL>` | Raw tool name to allow (repeatable). When set, only listed tools register. |
| `--disable-tool <TOOL>` | Raw tool name to block (repeatable). Applied after `--allow-tool`. |
| `--eager-load-tool <TOOL>` | Raw tool name to eager-load (repeatable). Listed tools skip the `tool_load` round-trip and ship in the cacheable tools-array prefix from turn 1. |
| `--deferral <MODE>` | Persist `deferral`: `deferred`, `eager` or `hidden`. Omitted, the server inherits `[mcp].deferral`. |
| `--tool-permission <TOOL=LEVEL>` | Per-tool permission override (repeatable). `LEVEL` is `none`, `read`, `workspace` or `unrestricted`. |
| `--no-login` | Skip the auto-login an HTTP server's probe would otherwise start; run `meka mcp login <name>` later. |
| `--required` | Persist `required = true`, so a turn is refused while this server is not connected. Omitted, the server inherits `[mcp].default_required` and is optional by default. |
| `--disabled` | Persist `disabled = true`, so the server is skipped entirely at startup. Re-enable with `meka mcp enable <name>`. |

### Example: Notion

These signposts are `info` logs, so they need `-v`; at the default `warn` level the command
succeeds silently and the exit code carries the result. Timestamps and targets are elided here.

```console
$ meka -v mcp add notion https://mcp.notion.com/mcp
added 'notion' to ~/.config/meka/config.toml
probe: 'notion' requires OAuth
running OAuth authorization for 'notion' (use --no-login to skip)
no [auth] block for 'notion'; assuming OAuth authorization_code
…
authorized 'notion'
```

`meka mcp add` on an HTTP endpoint:

1. **Probe**: issues an unauthenticated `GET` (3 s timeout, redirects off) and classifies the response per the MCP authorization spec + RFC 6750 + RFC 9728:

   - `2xx` → server is open, no login needed.
   - `401` / `403` with `WWW-Authenticate: Bearer …` → OAuth required. The `resource_metadata="…"` attribute (RFC 9728) is captured at DEBUG.
   - Any other status → couldn't infer, prints the status code.
   - Network failure → prints the error.

2. **Auto-login**: if the probe says OAuth is required (or `--auth oauth` was explicitly set), the OAuth authorization_code flow runs immediately as though the user had chained `meka mcp login <name>` themselves. The synthesized `[auth] = oauth` block is written back to `config.toml` on success.

3. **Rollback on failure**: if the OAuth flow errors out, the entry we just wrote is purged from `config.toml` (alongside any partial credentials), leaving the user's config clean. The command exits non-zero.

4. **`--no-login`**: skips step 2. The entry is still persisted and the probe's hint is still printed; run `meka mcp login <name>` when ready. Useful for scripted setup or when you expect to edit `[auth]` by hand.

The probe and the auto-login only run for HTTP servers, and only when the user didn't provide `--auth-token-stdin` (static bearer) or `--auth` (other than `oauth`). Stdio servers skip both.

Only `meka mcp login` runs the browser flow. A host that connects a server and finds no stored credential (the screen at startup, `meka serve`, `meka acp`, `/mcp reconnect`) marks it failed for want of OAuth credentials for its URL, which each surface words with its own login: `meka mcp login <name>` in a shell, `/mcp login <name>` on the screen, `POST /v1/mcp/{name}/login` over HTTP, rather than printing a URL to a terminal nobody may be watching.

### Remote hosts / SSH sessions

The OAuth flow redirects the browser to `http://127.0.0.1:<port>/callback`. When meka is running on a different host than the browser (SSH session, container, Codespace, WSL), the browser can't reach back and shows a "connection refused" error page. meka handles this automatically:

- While `meka mcp login <name>` waits for the callback it also watches stdin.
- The browser's address bar still contains the full callback URL (including `code` and `state`) even when the connection fails. Copy it, paste it into the meka prompt, and press Enter.
- Whichever completes first, the TCP callback or the pasted URL, wins.

meka prints the URL exactly once and leaves opening it to you, so the flow is the same on a desktop
and over SSH. The `authorized` line is an `info` log, shown here with `-v`.

```console
$ meka -v mcp login notion
To authorize, open this URL in your browser:

https://mcp.notion.com/authorize?response_type=code&…

Waiting up to 120s for the callback, or paste the callback URL here and press Enter:
http://127.0.0.1:46437/callback?code=…&state=…     ← paste here
authorized 'notion'
```

### Slash command parity

On the screen:
- `/mcp list` (or a bare `/mcp`): a picker of the configured servers with their state; Enter offers reconnect, login and logout for the picked one.
- `/mcp reconnect <server>`: reconnect smoke-test.
- `/mcp login <server>` / `/mcp logout <server>`: run the auth flow or revoke. Either ends by
  reading `config.toml` whole and serving it to the screen, as a write over HTTP does: the server
  restarts on the credential, or without it, and an account, a profile or a server edited in a
  shell since the screen opened is served with it. A file the screen could not start on refuses
  both until it is repaired. On the screen the URL and the paste field above are one question in
  the composer's place; Esc on it ends the login, and a paste the flow cannot read, or one carrying
  another login's state, is refused and asked again.
- `/mcp <server>:<prompt> [args...]`: render a server-defined prompt as the next user turn.

## Resources and prompts

In addition to tools, meka exposes MCP resources and prompts through several builtin tools (deferred: the agent finds them under `[Tool discovery]` or with `tool_search`, calls `tool_load` for the schema, then invokes them; absent on a `claude-subscription` profile, whose endpoint refuses tool names starting with `mcp_`; they follow [`[mcp].deferral`](../configuration/config-file.md#mcp-top-level-table), so `eager` declares them from the first request and `hidden` leaves them out):

| Builtin | Purpose |
|---------|---------|
| `mcp_resource_list` | List resources from one or every configured server. |
| `mcp_resource_read` | Read a resource by `server` + `uri`; text inline, binary base64-encoded. |
| `mcp_prompt_list` | List prompts from one or every configured server, including their declared arguments. |
| `mcp_prompt_get` | Render a prompt by `server` + `name` with optional `arguments`; returns `<role>: <text>` lines. |
| `mcp_resource_subscribe` | Subscribe to `resources/updated` notifications for a specific URI. Refused when the server will not watch it; otherwise the subscription outlives a reconnect. |
| `mcp_resource_unsubscribe` | Cancel a prior subscription. |
| `mcp_resource_updates_list` | List the latest recorded update per resource in the shared, bounded MCP context log; it resets when the host restarts. |

## Startup concurrency

MCP servers connect in parallel at startup, partitioned by transport so a fleet of stdio servers (process-spawn bound) doesn't fight a fleet of HTTP servers (network bound):

- stdio: [`[mcp].stdio_concurrency`](../configuration/config-file.md#mcp-top-level-table) (default `3`)
- http: [`[mcp].http_concurrency`](../configuration/config-file.md#mcp-top-level-table) (default `20`)

Both are tuning knobs: rarely needed, but useful if you're running ~30 stdio servers on a constrained box (lower it) or ~50 HTTP servers (raise it). `0` is refused at startup.

## Connection lifecycle

- **Protocol era**: every connect opens with a `server/discover` probe asking for MCP 2026-07-28 and falls back to `initialize` when the server does not answer as a modern one, the dual-era client the specification describes. The server's answer decides which era the connection speaks; `GET /v1/mcp` and the screen's `/mcp` picker show it. An HTTP server gets the fallback on the same connection, after a probe wait of up to ten seconds that [`[mcp].connect_timeout`](../configuration/config-file.md#mcp-top-level-table) does not shorten. A stdio server gets the probe on a child process of its own and, unless that child answers as a modern server within ten seconds (or half of `connect_timeout`, when that is less), a fresh child for the `initialize` handshake: a server from before the probe existed may swallow a method it does not know and answer nothing afterwards. Once that handshake has succeeded the verdict is kept for the life of the meka process, so a legacy stdio server pays the probe's wait once. A modern stdio server slower than the window to answer is given `initialize` instead: one that answers it is spoken to as legacy until meka restarts, one that does not fails its connect and is probed again on the retry.
- **Reconnection** is automatic for all transports (stdio, plain HTTP, OAuth-authenticated HTTP) when the transport closes mid-session. HTTP transports make five attempts 1s, 2s, 4s and 8s apart; stdio gets one immediate retry. The reconnect runs on a blocking thread to work around an upstream rmcp bug where the auth future is `!Send`. Every subscription the model made is made again on the new connection.
- **Exit**: a connect still in flight when meka exits is ended with it, and the child process it was spawning is killed, so a stdio server that ignores a closed stdin does not outlive meka that way.
- **Failed initial connect** is retried in the background with its own backoff (5s doubling to a 5 minute ceiling) until the server comes up, and the server's tools are registered into every live session when it does. A server that is slow to boot, or that starts after meka, therefore recovers on its own rather than staying `failed` for the life of the process. This matters most for a `required` server, where every turn is refused until it connects.
- **Session-expired recovery**: on a 2025-11-25 connection rmcp transparently re-initializes HTTP sessions on 404 / JSON-RPC `-32001`. meka relies on this; no per-call handling is required. A 2026-07-28 connection has no session to expire.
- **Cancellation**: when the agent cancels a tool call (e.g. Ctrl-C), meka sends `notifications/cancelled` to the server with the in-flight request id so the server can stop work.
- **Timeouts**: tool calls default to 10 minutes; override with `MEKA_MCP_TOOL_TIMEOUT`, a duration such as `20m`. A question a server asks mid-call ends with the call, however long the host would otherwise wait for an answer.
- **Tool list refresh**: on `tools/list_changed`, meka re-discovers the server's tools and hot-swaps them in the registry; no restart needed. A 2026-07-28 server sends nothing unsolicited, so meka opens one `subscriptions/listen` stream per connection asking for every list change and for the resources the model subscribed to; the stream is reopened whenever that set changes, and again, after a short backoff, when the server ends it or a proxy cuts it while the connection stands.
- **Progress notifications**: MCP tool calls attach a per-request `progressToken`; incoming `notifications/progress` render as a live status line under the tool invocation.
- **Call identity**: `tools/call` carries two extra keys in `_meta` alongside the progress token. `meka/sessionId` is the id (a UUID) of the session the call came from, letting a server scope per-session state (a cache, a workspace, an audit trail) to one conversation; a sub-agent reports its own child session id. `meka/toolUseId` is the provider's tool-use id for the call. Both are absent for calls made outside a session, such as connection-time handshakes.
- **Server instructions**: `InitializeResult.instructions` is captured once per connection and delivered in the per-turn context (sanitized and truncated to 2048 chars) under `[MCP server instructions]`. A server that connects late, or reconnects with different instructions, is announced as a change rather than rewriting anything already sent.
- **stdio server logs**: a stdio server's own stderr (many servers log there) is captured, not inherited, so it never corrupts the screen display. Each line is re-emitted on meka's `tracing` stream at `debug` level tagged with the server name, so it stays silent at default verbosity and surfaces under `-v` / `RUST_LOG`.
- `resources/list_changed`, `prompts/list_changed`, and `resources/updated` notifications are logged at `info`/`debug` level.

## Server-to-client features

| Feature | meka behavior |
|---------|----------------|
| `elicitation/create` | Routed to the calling session's frontend (screen / ACP form or URL prompt, or an HTTP session's `elicitation_required` event when it declared `supports_elicitation`) with a 60s timeout, when the server's in-flight tool calls all come from one session. Declined, with a warning, when no tool call is in flight, when the user does not answer in time, or when two sessions have calls in flight on the same server: the protocol ties an elicitation to nothing that names the call that provoked it, so routing it would be a guess. |
| `input_required` | How a 2026-07-28 server asks: inside the answer to `tools/call`, `resources/read` or `prompts/get`, instead of with a request of its own. An elicitation in a tool call's answer goes to the same prompt as above, and the call is sent again with the user's answer and the server's opaque state, up to ten rounds, all under the one tool-call timeout; an interrupt ends the wait for an answer as it ends the call. In a resource read or a prompt render there is no call to route it to, so it is declined. Sampling and roots are never asked for, because meka declares neither. |
