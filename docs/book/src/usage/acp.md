# ACP (Agent Client Protocol)

`agsh acp` speaks the [Agent Client Protocol](https://agentclientprotocol.com/) over stdio so editor / web / messenger clients can drive an agsh turn end to end. Where [Interactive Mode](./interactive-mode.md) and [One-Shot Mode](./one-shot-mode.md) are for humans, ACP is for *programs* that want to host agsh inside a richer UI — streamed diffs, native apply-buttons, hosted terminals, slash-command palettes.

This page describes what agsh's ACP surface looks like to a client. Editor-specific setup belongs in each editor's own documentation; the protocol contract is the same everywhere.

## Starting an ACP server

```bash
agsh acp
```

The process speaks JSON-RPC 2.0 with newline framing on stdio. There is no human-facing prompt — the binary is meant to be spawned by a client that owns the conversation. The client sends `initialize`, then `session/new` (or `session/load` / `session/resume`), then a series of `session/prompt` calls.

A few flags are worth knowing:

| Flag | Effect |
|------|--------|
| `-v` | Logs to stderr at `info` (incoming client identity, session lifecycle). |
| `-vv` | `debug` (per-request JSON-RPC diagnostics). |
| `RUST_LOG=agsh=trace` | Trace level. |

On startup, after the client's `initialize` arrives, agsh logs `ACP client connected: <name> <version>` so you can confirm the client identity in `-v` mode.

## What agsh advertises (`agentCapabilities`)

These are returned in `InitializeResponse.agentCapabilities`:

- **`loadSession: true`** — the client may call `session/load` with any persisted session id.
- **`sessionCapabilities.list`** — the client may call `session/list` to browse the persisted session catalogue (cwd-filtered, cursor-paginated; sub-agent audit sessions are hidden).
- **`sessionCapabilities.resume`** — the client may adopt a persisted session id without replaying history.
- **`sessionCapabilities.close`** — the client may release the active session slot.

`mcpCapabilities` is intentionally **not** advertised. agsh is itself an MCP client, but the servers it consumes are configured via agsh's own `config.toml` rather than the `mcpServers` field on `session/new`. Advertising HTTP/SSE while silently ignoring the client's array would have been misleading; the marker will return when client-supplied MCP server connections are actually implemented.

`agentInfo` carries agsh's name (`"agsh"`) and the running binary version.

## What agsh consumes (`clientCapabilities`)

The client advertises these in `InitializeRequest.clientCapabilities`; agsh stashes them and lets the built-in tools route accordingly:

- **`fs.readTextFile: true`** — `read_file` issues `fs/read_text_file { sessionId, path, line?, limit? }` so the client serves the *in-buffer* view of the file. Image and regex `read_file` modes have no `fs/*` analogue and stay local.
- **`fs.writeTextFile: true`** — `write_file` and `edit_file`'s apply step issue `fs/write_text_file { sessionId, path, content }`. agsh still attaches diff metadata to the `tool_call_update` so clients with an apply-diff UI can render it.
- **`terminal: true`** — `execute_command` runs the four-call dance: `terminal/create` → `terminal/wait_for_exit` → `terminal/output` → `terminal/release`. On `session/cancel` or a per-call timeout, agsh issues `terminal/kill` and still reads accumulated output. **Exception**: in `read` permission mode agsh keeps the local sandboxed shell (Landlock / bwrap / sandbox-exec / Low-Integrity) rather than delegating, so the sandbox isn't bypassed by the client's terminal.

If the client omits a capability, the matching tool falls back to local syscalls — the user-visible behaviour is the same as `agsh` in the REPL.

## Session lifecycle

agsh holds an in-memory map of `sessionId → SessionEntry`. Any number of sessions can coexist in one `agsh acp` process, each with its own cwd, permission level, conversation, cancellation token, and per-session runtime mutex. Prompts on different sessions run in parallel; a second `session/prompt` for a session that already has one in flight is rejected with `InvalidParams`. The session row is also locked on disk (the same lock the REPL uses), so two `agsh` processes can't simultaneously write events for the same session id.

- **`session/new { cwd, mcpServers }`** — mints a fresh persisted session, captures the cwd, takes the on-disk session lock, returns the session id and the current `SessionMode` state.
- **`session/load { sessionId, cwd, mcpServers }`** — replays the persisted conversation as a stream of `session/update` notifications (`user_message_chunk`, `agent_message_chunk`, `agent_thought_chunk`, `tool_call`, `tool_call_update`) before the response. Orphan tool calls (the persisted log stopped mid-tool) are closed out with a `failed` status so the client's UI doesn't render a stuck spinner. If the client's `cwd` differs from the persisted one, agsh updates the persisted row to match — the client wins.
- **`session/list { cwd?, cursor? }`** — paginated index. Filtered to the requested cwd when present; sub-agent sessions are always hidden. `nextCursor` is opaque — round-trip it back to keep paging.
- **`session/resume { sessionId, cwd, mcpServers }`** — adopts the session id without replaying. Use this when the client already has the history rendered. Same cwd-update behaviour as `session/load`.
- **`session/close { sessionId }`** — cancels any in-flight prompt, releases the on-disk session lock, and removes the entry from the map.
- **`session/cancel { sessionId }`** — interrupts the active `session/prompt`. The response carries `stopReason: "cancelled"`. If a cancel arrives between turns (after one prompt completed and before the next is installed), agsh latches the signal and cancels the next prompt immediately on arrival.
- **`session/set_mode { sessionId, modeId }`** — flips the agent's `Permission` cell. Modes outside `[permissions].enabled` from the config become JSON-RPC errors. On success, agsh emits `session/update: current_mode_update`. The flip is atomic and applies to the *next* tool call within an in-flight turn — no need to wait for the turn to finish.

## Prompt turn

A `session/prompt` carries a `prompt` array of `ContentBlock`s. agsh accepts `text` and `resource_link` blocks (the ACP baseline). `resource_link` blocks are flattened into a `<resource_link name="…" uri="…">description</resource_link>` tag inside the prompt text so the model can see the reference; agsh does not fetch the resource server-side. `image`, `audio`, and `resource` blocks are not yet supported and produce `InvalidParams`.

While the turn runs, agsh streams `session/update` notifications:

- `agent_message_chunk` for each piece of assistant text.
- `agent_thought_chunk` for thinking blocks (Claude OAuth / extended-thinking models).
- `tool_call` when a tool starts, with `kind`, `status: "in_progress"`, an absolute `locations` array (relative paths resolved against the session cwd), and the raw input.
- `tool_call_update` when a tool finishes, with the final `status` (`completed` / `failed`) and a `content` array. `edit_file` and `write_file` populate diff content blocks so clients can render the apply-diff UI.

The response carries a final `stopReason`:

| `stopReason` | Meaning |
|--------------|---------|
| `end_turn` | The agent finished cleanly. |
| `max_tokens` | The provider stopped because the model hit its maximum output tokens. The assistant message may be truncated. |
| `cancelled` | `session/cancel` interrupted the turn — including the case where the cancel caused an exception in an underlying operation. agsh probes the per-session cancellation token after `run_turn`; any error returned while the token has fired surfaces as `cancelled` rather than a generic JSON-RPC error. |
| `refusal` | The model declined to comply (Claude `stop_reason: "refusal"` and the OpenAI equivalent). The assistant message contains the refusal text. |

## Permission modes

agsh's `Permission` levels map 1:1 to ACP `SessionMode` ids:

| Permission | Mode id | Display name | Description |
|------------|---------|--------------|-------------|
| `None` | `none` | None | No tools available. |
| `Read` | `read` | Read | File reads and searches only. No writes, no shell. |
| `Ask` | `ask` | Ask | Every write or shell command requires approval. |
| `Write` | `write` | Write | All tools allowed without per-call approval. |

The full mode picker is advertised on every session-creation response (`NewSessionResponse.modes`, `LoadSessionResponse.modes`, `ResumeSessionResponse.modes`) but only the modes in `[permissions].enabled` from your `config.toml` are listed — picking a disabled mode would just error.

When the active mode is `ask`, write-gated tools trigger a `session/request_permission` round-trip. Clients render four options:

- **Allow** — run this call only.
- **Always allow** — run this call and skip the prompt for the same tool for the rest of the session.
- **Deny** — refuse this call only.
- **Always deny** — refuse this call and every subsequent call to the same tool.

Sticky decisions live in agsh's process memory; they reset on session close.

## Skills as slash commands

Every installed skill (see [Skills](./skills.md)) is advertised through `session/update: available_commands_update` after `session/new` / `session/load` / `session/resume`, and refreshed at the top of every `session/prompt` so a skill installed mid-session shows up without a reconnect.

Each command carries a generic free-form input hint (`"additional context (optional)"`). When the user picks one from the palette, the client typically inserts `/<skill-name> ` and lets the user type extra context. agsh parses the prompt as follows:

- Plain text (no leading slash) — passes through to the model unchanged.
- `/<skill-name>` matching an installed skill — loads the skill body via the same path as the REPL's `/skill` command (substituting `${AGSH_SKILL_DIR}` and `${AGSH_SESSION_ID}`) and prepends any extra context the user typed.
- Slash with a syntactically valid but unknown skill name (`/nonexistent`) — JSON-RPC error.
- Slash with content that isn't a valid skill identifier (`/etc/hosts`, `//comment`) — passes through to the model unchanged, so pasted paths and code comments don't get intercepted.

## Sub-agents

`spawn_agent` and skill-based delegation produce a sub-agent that runs through `PermissionForwardingFrontend`. The sub-agent's own output isn't streamed to the client (its final report flows back through the parent's `tool_call_update`), but its permission prompts, fs delegates, and terminal delegates all forward through the parent's connection — so the editor's apply-diff UI sees a sub-agent's writes the same as the main agent's.

## Known limitations

- **MCP `roots/list` from background queries.** During a tool call, `roots/list` reflects the calling session's cwd via a task-local override. Outside of a tool call (e.g. server-initiated polling), the handler falls back to the process cwd, since the MCP protocol doesn't carry session context.
- **Tool-call diff metadata isn't persisted.** A session reopened with `session/load` replays `tool_call_update`s as plain text rather than diffs. The on-disk content is unaffected.
- **`read` mode + `terminal` capability**: agsh runs the local sandboxed shell instead of delegating, to preserve the read-only jail. The shell appears in agsh's own output rather than the client's terminal pane until you switch to `ask` or `write`.
- **Image and regex `read_file`**: stay local. The `fs/read_text_file` request carries only text, so there's no protocol surface to delegate either case.
- **Single content type in prompts**: agsh's `session/prompt` accepts text only today. Image / audio / resource prompts will arrive as agsh's `PromptCapabilities` advertise them.
