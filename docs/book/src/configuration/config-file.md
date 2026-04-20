# Config File

agsh looks for a TOML configuration file at a platform-specific location:

| Platform | Path |
|----------|------|
| Linux | `~/.config/agsh/config.toml` (`$XDG_CONFIG_HOME/agsh/config.toml`) |
| macOS | `~/Library/Application Support/agsh/config.toml` |
| Windows | `%APPDATA%\agsh\config.toml` |

The config file is optional. If it does not exist, agsh silently skips it.

Set the `AGSH_CONFIG_DIR` environment variable to override the default location entirely — the value points at the `agsh` directory itself (contains `config.toml` and `skills/`). Useful for tests, portable installs, and isolating a per-project config from your global one.

## Format

```toml
[provider]
name = "openai"
model = "gpt-4o"
api_key = "sk-..."
base_url = "https://api.openai.com/v1"
```

All fields under `[provider]` are optional individually -- you can set some in the config file and override others with environment variables or CLI flags.

## Fields

### `provider.name`

The LLM provider to use.

| Value | Description |
|-------|-------------|
| `openai` | OpenAI Chat Completions API (also works with OpenAI-compatible APIs) |
| `claude` | Claude API (Messages endpoint) |

### `provider.model`

The model identifier to send to the provider. Examples:

- `gpt-4o`, `gpt-4o-mini` (OpenAI)
- `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` (Claude)
- Any model supported by an OpenAI-compatible endpoint

### `provider.api_key`

The API key for authentication. It is recommended to use environment variables (`OPENAI_API_KEY` or `CLAUDE_API_KEY`) instead of storing the key in the config file.

### `provider.oauth_token`

OAuth access token for the Claude provider. Can also be set via `CLAUDE_OAUTH_TOKEN` env var. The token is saved to the database on first use and loaded automatically on subsequent launches.

### `provider.oauth_token_url`

Custom OAuth token refresh endpoint. Defaults to `https://console.anthropic.com/v1/oauth/token`.

### `provider.base_url`

Custom API base URL. Useful for:

- Self-hosted models via [Ollama](https://ollama.ai) (`http://localhost:11434/v1`)
- [OpenRouter](https://openrouter.ai) (`https://openrouter.ai/api/v1`)
- Other OpenAI-compatible API providers

If not set, defaults to:

- `https://api.openai.com/v1` for the `openai` provider
- `https://api.anthropic.com` for the `claude` provider

### `provider.reasoning_effort`

Reasoning effort level for OpenAI o-series models. When set, the `reasoning_effort` parameter is included in API requests and `max_completion_tokens` is used instead of `max_tokens`.

Accepted values: `low`, `medium`, `high`. Omitted by default.

```toml
[provider]
reasoning_effort = "medium"
```

## Examples

### OpenAI

```toml
[provider]
name = "openai"
model = "gpt-4o"
# API key via env: export OPENAI_API_KEY=sk-...
```

### Claude

```toml
[provider]
name = "claude"
model = "claude-sonnet-4-20250514"
# API key via env: export CLAUDE_API_KEY=sk-ant-api03-...
# Or OAuth token via env: export CLAUDE_OAUTH_TOKEN=sk-ant-oat01-...
```

### Ollama (local)

```toml
[provider]
name = "openai"
model = "llama3"
api_key = "unused"
base_url = "http://localhost:11434/v1"
```

### OpenRouter

```toml
[provider]
name = "openai"
model = "anthropic/claude-sonnet-4-20250514"
base_url = "https://openrouter.ai/api/v1"
# API key via env: export OPENAI_API_KEY=sk-or-...
```

## `[display]`

Settings for output formatting.

### `display.render_mode`

Output render mode. Equivalent to the `--render-mode` CLI flag.

| Value | Description |
|-------|-------------|
| `bat` | Syntax-highlighted markdown via bat (default) |
| `termimad` | Terminal formatting via termimad (box-drawn code blocks, reflowed paragraphs). Alias: `rich` |
| `raw` | Raw markdown printed verbatim with aligned tables |

Default: `bat`

```toml
[display]
render_mode = "raw"
```

### `display.show_session_id_on_create`

Whether to display the session ID when a new session is created.

Default: `false`

### `display.show_session_id_on_exit`

Whether to display the session ID when agsh exits.

Default: `true`

```toml
[display]
show_session_id_on_create = true
show_session_id_on_exit = false
```

### `display.show_path_in_prompt`

Whether to show the current working directory in the interactive prompt.

Default: `true`

### `display.newline_before_prompt`

Whether to add a blank line before the prompt after each agent response.

Default: `true`

### `display.newline_after_prompt`

Whether to add a blank line after the prompt (before the agent response).

Default: `true`

```toml
[display]
show_path_in_prompt = false
newline_before_prompt = false
newline_after_prompt = false
```

## `[web]`

Settings for web-related tools (fetch_url, web_search).

### `web.user_agent`

Custom User-Agent string for HTTP requests. Some search engines may block requests with non-browser User-Agent strings.

Default: `Mozilla/5.0 (compatible; agsh/0.1)`

```toml
[web]
user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
```

## `[shell]`

Settings for shell command execution.

### `shell.sandbox`

Whether to enable read-only filesystem sandboxing for shell commands in read mode. When enabled (default), shell commands can be executed in read mode but with the filesystem physically write-protected. When disabled, shell commands require write mode.

Default: `true`

```toml
[shell]
sandbox = false  # disable sandboxed shell in read mode
```

The sandbox uses Landlock on Linux (kernel 5.13+) and sandbox-exec on macOS. On platforms where sandboxing is unavailable, shell commands always require write mode regardless of this setting.

## `[session]`

Settings for session history retention and context window management.

### `session.context_messages`

Maximum number of messages to send to the LLM API per request. Older messages are truncated from the beginning while preserving tool call chain integrity. The full history remains stored in SQLite -- only the API payload is limited.

Default: `200`

```toml
[session]
context_messages = 100
```

### `session.retention_days`

Automatically delete sessions older than this many days on startup. Uses the session's `updated_at` timestamp, so actively-resumed sessions are preserved even if originally created long ago.

Default: `90`

```toml
[session]
retention_days = 30
```

### `session.max_storage_bytes`

Maximum total byte size of all stored message content across all sessions. When exceeded on startup, the oldest sessions are deleted until the total is under the limit.

Default: `52428800` (50 MB)

```toml
[session]
max_storage_bytes = 10485760  # 10 MB
```

### `session.auto_compact`

Automatically compact the conversation when input tokens exceed 80% of the context window. Compaction summarizes older messages and preserves recent ones, the todo list, and scratchpad entries.

Default: `true`

```toml
[session]
auto_compact = false
```

### `session.context_window`

Override the model's context window size (in tokens). Used for auto-compact threshold calculation. If not set, agsh infers the context window from the model name.

```toml
[session]
context_window = 200000
```

## `[thinking]`

Settings for extended thinking (Claude provider only). Claude 4.6+ models use adaptive thinking automatically; older models use a fixed token budget.

### `thinking.enabled`

Whether to enable extended thinking. When enabled, the model can use additional tokens for internal reasoning before responding.

Default: `true`

### `thinking.budget_tokens`

Maximum number of tokens the model can use for thinking (for non-adaptive models).

Default: `16000`

```toml
[thinking]
enabled = true
budget_tokens = 20000
```

## `[prompt]`

Settings for injecting custom instructions into the system prompt. Use this to set installation-specific rules that should apply to every session -- things the agent needs to know about your system, preferred tools, or policies.

### `prompt.instructions`

A string of custom instructions that agsh will include in every system prompt, under a `## User Instructions` section. The model is told to treat them as hard constraints unless they conflict with safety requirements.

Suitable use cases:

- System-specific policies: "Never install Python packages globally with pip -- always use `uv` or a venv."
- Installed tooling the agent should know about: "Poppler is available on this system -- use `pdftotext` for PDFs."
- Workflow preferences: "Prefer ripgrep over grep; it's installed and faster."
- Signing / compliance rules: "Git commits on this system must use gpg signing."

Default: unset (no custom instructions).

```toml
[prompt]
instructions = """
Never install Python packages globally with pip. Always use `uv` or a venv.
Poppler is available on this system — use `pdftotext` for PDFs.
Prefer ripgrep over grep.
"""
```

Notes:

- Empty or whitespace-only strings are treated as unset.
- Instructions apply to sub-agents spawned via `spawn_agent` too.
- Instructions are included at all permission levels (including `none`) because they are authored by you.

## `[mcp]`

Settings for MCP (Model Context Protocol) tool servers. MCP allows agsh to discover and use tools provided by external servers.

### `[[mcp.servers]]`

An array of MCP server configurations. Each entry defines a server to connect to at startup.

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique name for this server. Used as namespace prefix for tools (`name__tool`). Must match `[A-Za-z0-9_-]+`, must not contain `__`, and must not be `agsh`, `ide`, or start with `mcp_`. |
| `transport` | Yes | Transport type: `"stdio"` (spawn subprocess) or `"http"` (streamable HTTP). |
| `command` | Stdio only | Path or name of the executable to spawn. On Windows, `npx` / `.cmd` / `.bat` / `.ps1` are auto-wrapped in `cmd /c`. |
| `args` | No | Arguments to pass to the command. |
| `env` | No | Environment variables to set for the spawned process (stdio only). |
| `url` | HTTP only | URL of the MCP server endpoint. |
| `auth_token` | No | Bearer token for HTTP authentication (sent as `Authorization: Bearer <token>`). |
| `auth` | No | OAuth authentication configuration (see below). Mutually exclusive with `auth_token`. |
| `headers` | No | Custom HTTP headers to include with every request (HTTP only). |
| `headers_helper` | No | Path to an executable whose stdout (`Name: Value\n` lines) is merged over `headers` at connect-time (HTTP only). Executed with `AGSH_MCP_SERVER_NAME` / `AGSH_MCP_SERVER_URL` in env; 15 s timeout. |
| `permission` | No | Server-wide permission override. Applies to every tool on this server, beating the `readOnlyHint` the server advertises and the `[mcp].default_permission` global fallback. See *Permission resolution* below. |
| `allowed_tools` | No | Optional allow-list of raw tool names (the form the server advertises, not the `server__tool` namespaced form). When set and non-empty, only these tools are registered; all others from this server are ignored. |
| `disabled_tools` | No | Optional block-list of raw tool names. Applied **after** `allowed_tools` — tools listed here are never registered. Both lists can coexist; the net set is `allowed_tools \ disabled_tools`. |
| `tool_permissions` | No | Per-tool permission overrides keyed by raw tool name. Beats the server-level `permission` and the server's `readOnlyHint` when resolving a tool's required permission. |
| `sampling` | No | Allow this server to call `sampling/createMessage` against your configured LLM provider. Default `false` (reject). Enabling this lets a compromised server inject arbitrary messages into your LLM context and burn your provider quota — opt in per-server, deliberately. |
| `sampling_limit` | No | Cap on sampling calls per agsh session from this server when `sampling = true`. Default `10`. Requests beyond the limit return an `INTERNAL_ERROR` to the server. |

### `[mcp]` top-level table

| Field | Purpose |
|-------|---------|
| `default_permission` | Fallback permission for MCP tools whose server didn't advertise `readOnlyHint` and doesn't have a `permission` override. Accepts `"none"`, `"read"`, `"ask"`, or `"write"`. If unset the hardcoded fallback is `"write"` (strict). |

### Permission resolution

Every MCP tool's required permission is resolved through a five-step chain; the first match wins:

1. **`server.tool_permissions[<raw-tool>]`** — explicit per-tool override.
2. **`server.permission`** — explicit server-level override. Applies to every tool on that server regardless of what the server advertises.
3. **`tool.annotations.readOnlyHint`** from the server: `true` → `Read`, `false` → `Write`.
4. **`[mcp].default_permission`** — global fallback.
5. **Hardcoded `Write`** — strict ultimate fallback.

User-supplied config (1, 2, 4) always beats the server's self-classification — if a server lies about a tool, you can override. But when no user config says anything, the server's hint is trusted for that specific tool so `readOnlyHint = false` destructive tools don't silently become Read-accessible just because the user opted into a lenient global default.

**Hint spoofing**: a compromised server could claim `readOnlyHint = true` on a destructive tool. Defend by setting `server.permission = "write"` on suspect servers (step 2 wins) or by listing the destructive tools explicitly in `tool_permissions` / `disabled_tools`.

**Stale config**: entries in `allowed_tools` / `disabled_tools` / `tool_permissions` that don't match any advertised tool get a `warn!` line at connect time. The server still connects; you just see a heads-up so you can clean up after the server renames a tool.

**Visibility across levels**: the resolved permission doesn't hide a tool from the agent. Every registered tool is listed in the system prompt with its required level noted inline, and a per-turn `[Permission context]` block names the current level plus any tools it blocks. The agent can still reason about an inaccessible tool and suggest `/permission <level>` to enable it; the permission gate is enforced at dispatch time. Keeping the tool catalogue visible across levels is also what lets the Anthropic prompt cache survive mid-session permission toggles.

#### Examples

Well-annotated server — no config needed. Every tool is classified by its own `readOnlyHint` (read tools Read, write tools Write):
```toml
[[mcp.servers]]
name = "notion"
transport = "http"
url = "https://mcp.notion.com/mcp"
```

User-declared trust on an unannotated server — all tools accessible in Read:
```toml
[[mcp.servers]]
name       = "internal"
transport  = "http"
url        = "https://mcp.internal/…"
permission = "read"
```

Overriding a mis-annotated or distrusted tool — one specific tool requires Write:
```toml
[[mcp.servers]]
name      = "notion"
transport = "http"
url       = "https://mcp.notion.com/mcp"

[mcp.servers.tool_permissions]
"notion-do-something-scary" = "write"
```

Subset of a server's tools — only `query` registers, all others are ignored:
```toml
[[mcp.servers]]
name          = "pg"
transport     = "stdio"
command       = "npx"
args          = ["-y", "@modelcontextprotocol/server-postgres"]
allowed_tools = ["query"]
```

Block-list with a narrow exception — all fs tools are Read-accessible except the two destructive ones, which are never registered:
```toml
[[mcp.servers]]
name           = "filesystem"
transport      = "stdio"
command        = "npx"
args           = ["-y", "@modelcontextprotocol/server-filesystem"]
permission     = "read"
disabled_tools = ["delete_file", "move_file"]
```

MCP tools are registered with namespaced names in the format `servername__toolname` to prevent collisions with built-in tools or between servers.

Tool and resource descriptions returned from MCP servers are truncated at 2048 characters to keep the system prompt bounded.

### Environment variable substitution

Every string field listed above (command, args, env values, url, headers values, auth_token) supports `${VAR}` and `${VAR:-default}` expansion from the process environment. Missing variables with no default leave the literal `${VAR}` in place and log a warning at startup. Use this to avoid committing secrets:

```toml
[[mcp.servers]]
name = "github"
transport = "http"
url = "https://mcp.github.com"
auth_token = "${GITHUB_MCP_TOKEN}"
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AGSH_MCP_TOOL_TIMEOUT` | `600000` ms (600 s) | Per-call timeout for MCP tools. Triggers `notifications/cancelled` on expiry. |

### `agsh mcp` CLI

Manage configured servers without editing `config.toml` by hand:

| Command | Action |
|---|---|
| `agsh mcp list` | Print all configured servers. |
| `agsh mcp get <name>` | Print full details for one server. |
| `agsh mcp add <name> <url-or-command> [args...] [flags]` | Persist a server. Transport is auto-detected: a URL starting with `http[s]://` means HTTP, anything else means stdio. Preserves existing formatting/comments via `toml_edit`. |
| `agsh mcp remove <name>` | Best-effort revoke stored OAuth tokens (RFC 7009) at the provider, then delete the server entry, clear stored credentials, and drop any resource-update ledger entries. |
| `agsh mcp reconnect <name>` | Smoke-test a connect; prints `ok` or the error. |
| `agsh mcp tools <name>` | Connect and list every advertised tool with its resolved permission, the chain step that decided it, and whether the current config allows it. Useful for populating `--allow-tool`, `--disable-tool`, or `--tool-permission` overrides without leaving the CLI. |
| `agsh mcp login <name>` | Drive interactive OAuth. If the server has no `[auth]` block and uses HTTP, assumes `type = "oauth"` and persists the block on success. |
| `agsh mcp logout <name>` | Call the provider's `revocation_endpoint` (RFC 7009) best-effort, then clear stored credentials + auth-probe cache. |

#### `agsh mcp add` flags

| Flag | Purpose |
|------|---------|
| `--transport <stdio\|http>` | Override the auto-detected transport. |
| `--env KEY=VALUE` | Environment variable for stdio (repeatable). |
| `--header KEY=VALUE` | HTTP header (repeatable). |
| `--auth <oauth\|client-credentials\|client-credentials-jwt>` | Configure the `[auth]` block. |
| `--auth-token <TOKEN>` | Static bearer token. Mutually exclusive with `--auth`. |
| `--client-id`, `--client-secret` | OAuth / client-credentials client identifiers. |
| `--signing-key <PATH>`, `--signing-algorithm <ALG>` | JWT signing material (`client-credentials-jwt` only). |
| `--scope <SCOPE>` | OAuth scope (repeatable). |
| `--redirect-port <PORT>` | Fixed OAuth redirect port (default: ephemeral). |
| `--permission <none\|read\|ask\|write>` | Per-server permission cap (applies to all tools on the server). |
| `--allow-tool <NAME>` | Raw tool name to allow (repeatable). When set, only listed tools register. |
| `--disable-tool <NAME>` | Raw tool name to block (repeatable). Applied after `--allow-tool`. |
| `--tool-permission <NAME=LEVEL>` | Per-tool permission override (repeatable). `LEVEL` is `none`/`read`/`ask`/`write`. |
| `--sampling`, `--sampling-limit <N>` | Opt into server-initiated `sampling/createMessage`. |

#### Example: Notion

```console
$ agsh mcp add notion https://mcp.notion.com/mcp
ok: added 'notion' to ~/.config/agsh/config.toml
probe: server requires OAuth.
running OAuth authorisation for 'notion' (use --no-login to skip).
no [auth] block for 'notion' — assuming OAuth authorization_code.
…
ok: authorized 'notion'
```

`agsh mcp add` on an HTTP endpoint:

1. **Probe** — issues an unauthenticated `GET` (3 s timeout, redirects off) and classifies the response per the MCP authorization spec + RFC 6750 + RFC 9728:

   - `2xx` → server is open, no login needed.
   - `401` / `403` with `WWW-Authenticate: Bearer …` → OAuth required. The `resource_metadata="…"` attribute (RFC 9728) is captured at DEBUG.
   - Any other status → couldn't infer, prints the status code.
   - Network failure → prints the error.

2. **Auto-login** — if the probe says OAuth is required (or `--auth oauth` was explicitly set), the OAuth authorization_code flow runs immediately as though the user had chained `agsh mcp login <name>` themselves. The synthesised `[auth] = oauth` block is written back to `config.toml` on success.

3. **Rollback on failure** — if the OAuth flow errors out, the entry we just wrote is purged from `config.toml` (alongside any partial credentials + probe cache), leaving the user's config clean. The command exits non-zero.

4. **`--no-login`** — skips step 2. The entry is still persisted and the probe's hint is still printed; run `agsh mcp login <name>` when ready. Useful for scripted setup or when you expect to edit `[auth]` by hand.

The probe and the auto-login only run for HTTP servers, and only when the user didn't provide `--auth-token` (static bearer) or `--auth` (other than `oauth`). Stdio servers skip both.

#### Remote hosts / SSH sessions

The OAuth flow redirects the browser to `http://127.0.0.1:<port>/callback`. When agsh is running on a different host than the browser (SSH session, container, Codespace, WSL), the browser can't reach back and shows a "connection refused" error page. agsh handles this automatically:

- While `agsh mcp login <name>` waits for the callback it also watches stdin.
- The browser's address bar still contains the full callback URL (including `code` and `state`) even when the connection fails. Copy it, paste it into the agsh prompt, and press Enter.
- Whichever completes first — the TCP callback or the pasted URL — wins.

```console
$ agsh mcp login notion
server 'notion' has no [auth] block; assuming OAuth authorization_code.
Opening browser for MCP server 'notion' OAuth authorization...
If the browser didn't open, visit:
  https://mcp.notion.com/authorize?response_type=code&…
Waiting for OAuth callback (up to 120s).
  If the browser can't reach this host (e.g. you're over SSH), paste the full
  callback URL here and press Enter.
http://127.0.0.1:46437/callback?code=…&state=…     ← paste here
ok: authorized 'notion'
```

#### REPL parity

Inside the REPL:
- `/mcp list` — list configured servers.
- `/mcp reconnect <server>` — reconnect smoke-test.
- `/mcp login <server>` / `/mcp logout <server>` — run the auth flow or revoke.
- `/mcp <server>:<prompt> [args...]` — render a server-defined prompt as the next user turn.

### Resources and prompts

In addition to tools, agsh exposes MCP resources and prompts through four builtin tools (deferred — the agent activates them when needed):

| Builtin | Purpose |
|---------|---------|
| `list_mcp_resources` | List resources from one or every configured server. |
| `read_mcp_resource` | Read a resource by `server` + `uri`; text inline, binary base64-encoded. |
| `list_mcp_prompts` | List prompts from one or every configured server, including their declared arguments. |
| `get_mcp_prompt` | Render a prompt by `server` + `name` with optional `arguments`; returns `<role>: <text>` lines. |
| `subscribe_mcp_resource` | Subscribe to `resources/updated` notifications for a specific URI. |
| `unsubscribe_mcp_resource` | Cancel a prior subscription. |
| `list_mcp_resource_updates` | Print every resource that has been reported as updated since the session started. |

### Connection lifecycle

- **Reconnection** is automatic for all transports (stdio, plain HTTP, OAuth-authenticated HTTP) when the transport closes mid-session. HTTP transports use exponential backoff (1s, 2s, 4s, 8s, 16s, capped 30s, max 5 attempts); stdio gets one immediate retry. The reconnect runs on a blocking thread to work around an upstream rmcp bug where the auth future is `!Send`.
- **Session-expired recovery**: rmcp 1.5 transparently re-initialises HTTP sessions on 404 / JSON-RPC `-32001`. agsh relies on this; no per-call handling is required.
- **Cancellation**: when the agent cancels a tool call (e.g. Ctrl-C), agsh sends `notifications/cancelled` to the server with the in-flight request id so the server can stop work.
- **Timeouts**: tool calls default to 600 s; override with `AGSH_MCP_TOOL_TIMEOUT` in ms.
- **Tool list refresh**: on `tools/list_changed`, agsh re-discovers the server's tools and hot-swaps them in the registry — no restart needed.
- **Progress notifications**: MCP tool calls attach a per-request `progressToken`; incoming `notifications/progress` render as a live status line under the tool invocation.
- **Server instructions**: `InitializeResult.instructions` is captured once per connection and spliced into the system prompt (sanitised + truncated to 2048 chars) under `## MCP Server Instructions`.
- **Auth-probe cache**: 401 responses are cached for 15 minutes so a restart after a failed auth flow skips the unauthenticated probe and goes straight to OAuth. Cleared by `agsh mcp logout`.
- `resources/list_changed`, `prompts/list_changed`, and `resources/updated` notifications are logged at `info`/`debug` level.

### Server-to-client features

| Feature | agsh behaviour |
|---------|----------------|
| `roots/list` | Returns a single root: `file://<current-working-directory>` with the directory basename as the name. |
| `elicitation/create` | Always responds with `Decline` and logs a warning — interactive form/URL input is not wired into the REPL. |
| `sampling/createMessage` | Rejected with `METHOD_NOT_FOUND` unless the server has `sampling = true` in its config. When allowed, the current provider handles the request; per-session `sampling_limit` caps how many times each server may invoke it. |

### `[mcp.servers.auth]`

OAuth authentication for HTTP MCP servers. Set `type` to choose the authentication method. This is mutually exclusive with `auth_token`.

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Auth method: `"client_credentials"`, `"client_credentials_jwt"`, or `"oauth"` |
| `client_id` | Varies | OAuth client ID (required for client_credentials/jwt, optional for oauth with dynamic registration) |
| `client_secret` | Varies | Client secret (required for client_credentials, optional for oauth) |
| `scopes` | No | OAuth scopes to request |
| `resource` | No | Resource parameter ([RFC 8707](https://datatracker.ietf.org/doc/html/rfc8707)), client_credentials only |
| `signing_key_path` | JWT only | Path to PEM private key file |
| `signing_algorithm` | No | JWT signing algorithm: `RS256` (default), `RS384`, `RS512`, `ES256`, `ES384` |
| `redirect_port` | No | Local port for OAuth authorization code callback. When omitted, agsh binds to a random ephemeral port (recommended). `oauth` only. |

### Examples

#### Stdio server

```toml
[[mcp.servers]]
name = "postgres"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
permission = "write"
```

#### HTTP server

```toml
[[mcp.servers]]
name = "web-tools"
transport = "http"
url = "http://localhost:8080/mcp"
permission = "read"
```

#### HTTP server with authentication

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"
auth_token = "your-bearer-token"
permission = "write"

[mcp.servers.headers]
X-Custom-Header = "value"
```

#### Stdio server with environment variables

```toml
[[mcp.servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
permission = "read"

[mcp.servers.env]
GITHUB_TOKEN = "ghp_..."
```

#### Multiple servers

```toml
[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
permission = "read"

[[mcp.servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
permission = "write"
```

#### HTTP server with OAuth client credentials

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"
permission = "write"

[mcp.servers.auth]
type = "client_credentials"
client_id = "my-client-id"
client_secret = "my-client-secret"
scopes = ["read", "write"]
```

#### HTTP server with JWT client credentials

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials_jwt"
client_id = "my-client-id"
signing_key_path = "/path/to/private-key.pem"
signing_algorithm = "RS256"
scopes = ["admin"]
```

#### HTTP server with OAuth authorization code flow

On first connection, agsh opens a browser for authorization and stores the token for future use.

```toml
[[mcp.servers]]
name = "github-mcp"
transport = "http"
url = "https://mcp.example.com"

[mcp.servers.auth]
type = "oauth"
client_id = "my-app-id"
scopes = ["repo", "user"]
redirect_port = 8400
```

If `client_id` is omitted, agsh attempts [dynamic client registration](https://datatracker.ietf.org/doc/html/rfc7591) with the server.
