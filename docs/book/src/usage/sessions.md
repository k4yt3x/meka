# Sessions

Sessions persist your conversation so you can resume later. Each session has an id and lives in the store, a SQLite file.

## How sessions work

- A session is **not** created when meka starts. It is created lazily when you send the first message.
- When a session is created, its id is printed to stderr.
- When you exit meka (Ctrl+D), the session id is printed again so you can note it for later.
- Sessions include the full conversation: your inputs, the agent's responses, and tool call results.

## Resuming a session

### Continue the last session

```bash
meka -c
```

This resumes the most recently updated session. `-c` takes no value; give an opening prompt with `-p`: `meka -c -p "and now add tests"`.

### By id

```bash
meka -r 550e8400-e29b-41d4-a716-446655440000
```

The agent loads the previous conversation and continues from where you left off.

### By id prefix

If the value passed to `-r` isn't a whole id, meka treats it as a leading prefix and looks up sessions whose id starts with it. This avoids having to copy the entire id:

```bash
meka -r 550e            # works if exactly one session starts with `550e`
meka -r 5               # likely ambiguous; meka lists matching ids and exits
```

When a prefix matches multiple sessions, meka prints the matching ids (most-recent first) so you can disambiguate. Type a few more characters until the prefix is unique.

### What a resume restores

A session records what it runs on, and a resume brings all of it back:

- **The profile.** A session started with `--profile openai` resumes on `openai`, whatever
  `default_profile` says. This matters beyond the surprise: a thinking block is tagged with the
  backend that produced it and is not replayed to a different one, so resuming on another profile
  could silently discard the reasoning the conversation recorded, and a different account would be
  billed.
- **The permission level, and the approvals switch.** A session created at `unrestricted` resumes there without the flag, and one that had `/approvals on` comes back asking.

Everything the profile and its account state comes with it: the model, the endpoint, the context
window the gauge and auto-compaction measure against, and whether images may be attached. A session
records the profile's *name*, not a copy of its settings, so editing the profile with
[`meka profile set`](../configuration/config-file.md#meka-profile-cli) moves every session on it.
Two sessions on one `meka serve` can sit on profiles with different windows and each is measured
against its own.

Naming `--profile` on a resume **repins the session**: the row is rewritten, so it keeps that
profile from then on rather than for one run. `--permission` repins the same way.

You can also change the profile mid-session: `/profile <name>` in the REPL,
`PATCH /v1/sessions/{id}` with `{"profile": "..."}` over HTTP, or the Profile picker in an ACP
client. Each rewrites the row.

That `PATCH` is also how you rescue a session over HTTP when its profile has left `config.toml`: a
body naming only a profile moves the row without building an agent for it, so it works on a session
that cannot currently run. From the CLI the equivalent is `meka -r <id> --profile <name>`.

Switching profile mid-conversation is allowed and is your call. From the next turn the model no
longer sees the reasoning recorded under the old profile, for the reason above.

### What a resume does not restore

`--writable-root` is not restored, because it belongs to the process rather than the session; pass
it again. See [permissions](./permissions.md) for why recording it would be wrong.

**A resumed session opens in the directory it recorded**, not the one your shell is in. `meka -c` from anywhere reopens the session where it was working, and `/cd` is the only thing that moves it. This is deliberate: at [`workspace`](./permissions.md) the working directory *is* the writable boundary, so adopting the shell's would silently widen it: resume a project session from `$HOME` and the whole home directory becomes writable, with a [scheduled job](./scheduling.md) able to fire before you could react. If the recorded directory has since been removed, meka warns and opens where you are. To get back to your shell's directory, run `/cd` with no argument.

A resume restores the conversation, not the world it ran in. The messages come back verbatim, which means the agent reads its own earlier tool calls and can reasonably assume their effects still hold. Two kinds of state do not survive the process that made them:

- **Which files have been read.** meka tracks reads in memory so `edit_file` can refuse to write over a file the agent has not seen. A new process starts with that record empty, so the first edit to any file asks for a `read_file` first.
- **Anything an MCP server was holding.** A loaded database, an authenticated session, a subscription: these belong to the server's process, not to the conversation, and a reconnect drops them. meka has no way to model what a given server keeps open.

Everything else is restated in the per-turn context on every turn regardless (permission level, working directory, todo list, tool catalog), and background tasks that were running deliver an `interrupted` outcome, so none of those can go stale unnoticed.

Because the second kind is unknowable from meka's side, the first turn after a resume carries a `[Session resumed]` note telling the agent to re-establish rather than assume. It appears once and is not repeated. There is nothing to configure.

## Session locking

Only one meka instance can be attached to a session at a time. This prevents race conditions from concurrent writes.

- The lock is taken the moment the session row exists, which for a brand-new session is at the start of its first turn rather than the end. A second invocation launched while that turn is still running is refused like any other.
- If you try to resume a session that is locked by a running meka process, you will get an error.
- If the locking process has exited (crashed or was killed), meka detects this and allows you to take over the lock.
- Under ACP (`meka acp`), the lock is released as soon as the editor disconnects: closing the connection (stdin EOF) or sending SIGTERM/Ctrl-C makes `meka acp` exit, so the session can be reopened immediately.

## Storage location

Sessions live in the store, a SQLite file at a platform-specific location:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/meka/meka.db` (`$XDG_DATA_HOME/meka/meka.db`) |
| macOS | `~/Library/Application Support/meka/meka.db` |
| Windows | `%APPDATA%\meka\meka.db` |

### What else is in that directory

`meka.db-wal` and `meka.db-shm` are SQLite's own companions to the open store, not backups. The
`locks/` subdirectory holds the lock files meka uses to keep two processes off one session and off
one schema change; it is empty of anything worth reading.

You may also find one `meka.db.v<version>.bak`. meka copies the store aside before it changes the
schema, and keeps exactly one such copy: the next schema-changing upgrade takes a fresh one and
removes the one before it, so the copies do not accumulate. Expect the data directory to settle at
roughly twice the size of the store, and to peak higher than that during an upgrade, since the new
copy is written before the old one goes.

To restore one, stop every meka process and copy it over `meka.db`. It records the schema version it
was taken at, so the next start brings it forward again rather than mistaking it for a current store.
Because only the newest is kept, restoring undoes the most recent schema-changing upgrade and
nothing before it; move a copy of your own aside if you want to go back further.

An interrupted upgrade can leave a `meka.db.v<version>.bak.partial` behind. That is a copy that never
finished, so it is not restorable and nothing removes it; delete it whenever you like.

Anything else you put in this directory is yours and meka leaves it alone, including a file whose
name merely resembles the above.

## Store schema

The three tables below are the conversation itself. The store holds seven more, which the
features that own them document: `scheduled_jobs` ([scheduling](./scheduling.md)), `background_tasks`
([background work](./background.md)), `memories` and its `memories_fts` full-text index
([memory](./memory.md)), `prompt_history` (the REPL's
[input history](./interactive-mode.md#input-history)), and `account_credentials` and
`mcp_credentials` (secrets, never in `config.toml`).

**sessions**, one row per session:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Primary key |
| `created_at` | TEXT (RFC 3339) | When the session was created |
| `updated_at` | TEXT (RFC 3339) | When the session was last updated |
| `parent_session_id` | TEXT (UUID) | The session that spawned this sub-agent, or NULL |
| `cwd` | TEXT | Working directory the session is in; moved only by `/cd`, ACP, or `PATCH` |
| `permission` | TEXT | Permission level a re-attached session resumes with, and the level a scheduled gate is re-checked against |
| `approvals` | INTEGER | Whether calls above the level are put to the user for approval |
| `capabilities_json` | TEXT | Per-session capability flags, for HTTP re-attach |
| `token_id` | TEXT | Bearer token that created the session, for HTTP |
| `additional_roots_json` | TEXT | Workspace roots beyond `cwd` |
| `subagent_spec_json` | TEXT | The terms a sub-agent was spawned under |
| `stat_*` | INTEGER | Eight cumulative counters behind `/status` |
| `profile` | TEXT | Profile the session runs on. Never NULL, though a row carried forward from a store that predates the column can hold `''` |

**blobs** and **message_blobs**: image bytes by SHA-256 content hash, and which message rows
reference them. A message row holds a reference in place of the bytes, so a screenshot read twice is
stored once, and deleting a session removes the blobs nothing else references. An export carries the
bytes its sessions reference, and `GET /v1/sessions/{id}/blobs/{hash}` serves them over HTTP.

Locks are OS file locks under the data directory, not a column: a row cannot record a crashed
process's PID and lock a session forever.

**messages**, one row per message in a session:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing primary key |
| `session_id` | TEXT (UUID) | Foreign key to `sessions.id` |
| `role` | TEXT | `user_blocks` (a turn: its `turn_context` and `text` blocks and any images), `user` (a plain text message meka wrote), `assistant`, `tool_results`, `compact_boundary`, `repair`, or `redact` |
| `content` | TEXT | Message content (plain text or JSON) |
| `created_at` | TEXT (RFC 3339) | When the message was saved |

**tool_outputs**, scratchpad entries, one row per entry:

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT (UUID) | Part of composite primary key |
| `name` | TEXT | Part of composite primary key |
| `content` | TEXT | The stored content |
| `created_at` | TEXT (RFC 3339) | When the entry was created |

Scratchpad entries are scoped to a session. Two sessions can have entries with the same name. Entries are preserved across compaction but deleted when a session is deleted.

## History retention

**meka never deletes sessions unless you ask it to.** Conversation history isn't reproducible, so there is no default cleanup by age and none at all by size.

If you do want a time window, set it explicitly:

```toml
[session]
retention = "30d"   # delete sessions not updated in 30 days, at startup
```

With that set, meka deletes matching sessions when the agent starts and says so at `warn` level, so a deletion you configured is still a deletion you see. A session another meka process has open is spared, and so is the one you are resuming with `-c` or `-r`, however old it is. Unset (the default) keeps everything forever.

To prune on demand instead, delete on your own schedule:

```bash
meka session delete --older-than-days 90   # same window, run when you choose
meka session delete <id> [<id>…]           # specific sessions
meka session delete --all                  # everything
```

Deleting a session also removes its messages, scratchpad entries, and any sub-agent children.

A session is locked from the moment it exists: the lock is taken before the row is written, so a sweep in another terminal cannot catch it in between. That holds for new sessions, for sub-agent sessions, for forks made from any surface, and for the root of an imported archive while it is written. Copying a conversation holds it still too: every fork door (`meka session fork`, `/fork`, `POST /v1/sessions/{id}/fork`, ACP `session/fork`) and `meka session export` refuse a session another process has open, because a copy taken mid-turn ends on a user message the model never answered and restores as an unusable session. `meka session rewind` has always done this.

No deletion touches a session another meka process has open. Naming one by id fails and says so; `--all`, `--older-than-days` and the startup sweep skip it and report how many they left behind. This matters most for the startup sweep, because only turns bump a session's timestamp (resuming does not), so a REPL left at its prompt past the window looks expired while somebody is sitting in front of it.

See [Config file](../configuration/config-file.md#session) for details.

## Context window limiting

Long sessions can exceed the LLM's context window or become expensive. The `context_messages` setting (default: `200`) limits how many recent messages are sent to the API:

```toml
[session]
context_messages = 100
```

The full history remains in the store for resumption. Only the API payload is truncated. The cap applies to every request in a turn, not just the first, so a long tool loop cannot grow the payload past it mid-turn, and the truncation preserves tool call chains (it never splits a tool use from its result). Removing the key restores the default of `200` rather than lifting the cap.

The tool catalog and skill list travel in the conversation rather than the system prompt, so they are subject to this window too. meka tracks where it last stated them and restates them in full once that message scrolls out, which works out to roughly once per window. Setting `context_messages` very low therefore makes those restatements more frequent.

### Compacting a session

When a session becomes too long, `/compact` replaces the older turns with a summary and keeps a token-budgeted tail of the most recent messages verbatim (snapped to a clean user-turn boundary so tool calls aren't split).

By default the summary is written by **the agent itself**, in a *checkpoint turn* that runs before anything is discarded. The agent gets its real system prompt, its memory index, the full conversation, and a small set of tools, and is told its context is about to be replaced. It saves whatever must outlive the window (`memory_write` for facts and decisions that should still be true in a future session, the scratchpad for working material), then calls `context_replace` with the summary.

This matters because compaction is the one moment information is destroyed, and before this it was also the one moment the agent could not act. The alternative, a separate summarizer call, knows nothing about who the agent is or what it is for.

A checkpoint can **save, but not act**. It reaches the memory, scratchpad, todo, conversation-history and read-only search tools, and nothing else: no shell, no file writes, no sub-agents, no scheduling, no MCP. The delete tools are excluded too, since deleting is not saving and a mistaken delete in an unattended checkpoint is unrecoverable. A tool disabled in `[tools]` stays disabled here.

You can say what to keep:

```text
/compact keep the auth refactor decisions, drop the debugging
```

The confirmation reports what was written, because memories are durable and instance-scoped:

```text
Session compacted. Wrote 2 memories: deploy-pipeline-quirks, api-rate-limits.
```

Note that an *automatic* compaction runs a checkpoint too, unattended, and can write memory without anyone watching.

Compaction preserves scratchpad entries and the todo list, and re-injects environment context so the agent isn't disoriented afterwards. The tool catalog, skill list, and MCP server instructions are restated in full on the next turn, since the messages that carried them may have been summarized away. Tools loaded via `load_tool` stay loaded; the deferred-tool active set is snapshotted into the compaction boundary. If a detail was dropped, the model can `conversation_search` / `conversation_read` the full pre-compaction history, which stays on disk.

Internally, compaction does not delete pre-compaction rows from the store. It appends a `compact_boundary` row to the `messages` table; the materialized view is reconstructed from the event log, so the persisted log itself stays append-only.

#### When the summarizer runs instead

A standalone summarizer, with no tools and none of the agent's identity, is the fallback. It runs when:

- The compaction is an **emergency** one, i.e. the provider has already rejected the request for exceeding the window. A checkpoint turn re-sends that same conversation, so it would be refused identically; the summarizer strips images and truncates long blocks, which is what lets it get through.
- The checkpoint turn **fails or produces nothing usable**.
- `compact_checkpoint` is off.

There is one rung in between: if the checkpoint turn ends without calling `context_replace` but did write a summary in prose, that text is used. `tool_choice` isn't available across meka's backends, so the call can't be forced.

```toml
[session]
compact_checkpoint = true   # default
```

Turning it off leaves the standalone summarizer to write every summary, which saves one model call per compaction at the cost of the agent having no say in what survives.

### Auto-compact

When `auto_compact` is enabled (default: `true`), meka automatically compacts the conversation when the input token count exceeds 80% of the context window. The threshold check runs between turns, not during tool loops. It is both reactive (the previous turn's reported usage) and proactive (an estimate of the next request, so a turn whose own input jumps over the window is compacted before it is sent). As a last resort, if the provider still rejects a request for exceeding the context window, meka compacts once and retries the turn instead of failing.

```toml
[session]
auto_compact = true
context_window = 200000  # optional override
```

### Agent-initiated compaction

The agent doesn't have to wait for the threshold. `context_compact` asks for a compaction before the agent's next step: it runs once the current batch of tool calls finishes, and the turn then carries on against the summary. What it reclaims is history from earlier turns: with the default `keep_recent`, the tail is cut back to a clean user boundary, so the current turn stays verbatim and an agent that filled its window with this turn's own tool results gets little back. One compaction per turn: a further request once the first has run is ignored, and the agent can ask again on a later turn.

```text
context_compact(instructions: "the day's work is in memory now", keep_recent: false)
```

`keep_recent: false` skips the verbatim tail entirely, so the summary is all that remains. That is the difference between compacting and turning the page, and it's what makes a "start of a new day" routine work: a scheduled job at midnight can write the day's diary to memory, then compact clean, instead of carrying yesterday's context forward indefinitely.

The request is parked rather than applied where it is made: a tool cannot rewrite the conversation the agent loop is holding. It is drained at the next boundary between rounds, once the batch's tool results are in, which is what lets the rest of the turn run against the summary.

### What the agent sees

Once a turn has been measured, the per-turn context block carries a `[Context budget]` line reporting occupancy and the threshold compaction fires at:

```text
[Context budget]
Using ~84k of 200k tokens (42%). The conversation is summarized automatically at
80%, which loses detail, so prefer to finish or checkpoint work before then.
```

The agent is expected to budget its own reading and to decide when a task will fit, so it needs the same number the harness uses. Without it, those are guesses. The line is suppressed when the window is unknown, and on the first turn of a session, when there is no measurement yet rather than a genuine zero.

It rides the per-turn context block rather than the system prompt because it changes every turn and the system prompt is the cached prefix.

From the second compaction onward the line also reports how many have happened, since a summary of a summary has lost considerably more than a first pass:

```text
This conversation has been summarized 3 times, so early detail is now several
removes from what was said; write anything that must last to memory rather than
relying on it surviving another pass.
```

Because that block is rendered once per turn, it does not move while the agent works. During a long tool loop, which is exactly when context moves fastest, it is stale. `context_check` reports the live figures on demand: occupancy, headroom in tokens, the fixed overhead compaction cannot reclaim, how much of the recent conversation would survive verbatim, and the compaction count. Refreshing the pushed block instead would rewrite a message the provider's prompt cache already covers, invalidating it on every iteration; a tool result appends at the tail and is cache-safe.

## Listing sessions

To see past sessions:

```bash
meka session list
```

This shows a table with each session's id, last update time (local, with its UTC offset), profile,
and its title, the words of the first message:

```
ID        Updated                  Profile   Title
550e8400  2026-03-14 12:00 +00:00  work      How do I implement a binary search tree?
a1b2c3d4  2026-03-13 09:30 +00:00  personal  Fix the login page CSS
```

The `ID` column shows as much of each id as distinguishes it from the others on screen, widening only
if two would otherwise read the same. Every command that takes a session id (`meka -r`, `export`,
`show`, `fork`, `rewind`, `delete`) accepts any unique prefix, so what you see is normally what
you retype. An ambiguous prefix is refused and every match listed, rather than acted on.

Uniqueness is computed over the rows *on screen*, while the commands resolve against every session
in the store. A listing narrowed by `-n`, or one hiding sub-agent sessions (they are hidden unless
`--include-children` is given), can therefore print a prefix that a wider set makes ambiguous. That
fails closed: the command refuses and names both ids, so nothing is acted on and the full id is one
copy away.

For the whole id, and the working directory and permission the table has no room for:

```bash
meka session show 550e8400
```

By default the 20 most recent sessions are shown. Use `-n` to change:

```bash
meka session list -n 50
```

Sub-agent transcripts are hidden by default, so the listing stays the conversations you started. Add
`--include-children` to see them too:

```bash
meka session list --include-children
```

The Profile column names the profile, which is the whole story: a session records a profile name
and nothing else, so the model and endpoint it runs on are whatever that profile and its account
currently say. `meka profile list` shows them.

Both commands take `--format json`. The listing becomes `{"sessions": [...]}` and `show` one object,
each session carrying `id`, `created_at`, `updated_at`, `profile`, `title`, `approvals`, and, when
the row records them, `cwd`, `permission`, `capabilities` and `parent_id`: the fields
[`GET /v1/sessions`](http-api.md) returns under the same names, less the two only a running host
can answer (`turn_in_flight`, `last_turn_at`). Ids are printed in full, and an empty store is
`{"sessions": []}`.

```bash
meka session list --format json | jq -r '.sessions[] | "\(.id) \(.profile)"'
meka session show 550e8400 --format json | jq .cwd
```

## Exporting a session

You can export any session as a Markdown file:

```bash
meka session export 550e8400-e29b-41d4-a716-446655440000
```

This writes `session-550e8400-e29b-41d4-a716-446655440000.md` in the current directory with the full conversation. User and assistant messages are rendered as Markdown sections, while tool calls and results are wrapped in collapsible `<details>` blocks. The export always covers the **entire** session, including turns that were later hidden from the model by [compaction](interactive-mode.md#compact) (each compaction point is marked with its summary).

To write to a specific file:

```bash
meka session export 550e8400-e29b-41d4-a716-446655440000 -o conversation.md
```

To print to stdout (for piping):

```bash
meka session export 550e8400-e29b-41d4-a716-446655440000 -o -
```

### JSON (structured, round-trippable)

Pass `--format json` for a structured export instead of rendered Markdown:

```bash
meka session export 550e8400-e29b-41d4-a716-446655440000 --format json
```

This writes `session-<id>.json`, a lossless dump of the session's event log (including input images and compaction boundaries), its cumulative stats, and scratchpad entries. The archive carries `format_version: 2`, and an import refuses any other version rather than guessing at its shape. Unlike Markdown, a JSON export also includes any **sub-agent child sessions** spawned during the conversation, and it can be re-imported with `meka session import`. It deliberately contains **no credentials**: API keys and OAuth tokens live in separate tables and are never part of an export.

## Importing a session

Recreate a session from a JSON export:

```bash
meka session import session-550e8400-e29b-41d4-a716-446655440000.json
```

meka assigns the imported session (and any sub-agent children) **new** ids so they can't collide with existing sessions, then prints the new root session id. Resume it like any other session:

```bash
meka -r <new-id>
```

Read from stdin with `-`:

```bash
cat session.json | meka session import -
```

The import preserves the full conversation, per-message timestamps, cumulative stats, scratchpad entries, and the name of the profile the session ran on. That name is all an archive carries about the profile: the settings themselves come from whatever `[profiles.<name>]` and its account say on the installation importing it. An archive that names no profile adopts this installation's default instead; repin it with `--profile` if it ran somewhere else. If nothing can supply one, because no `default_profile` is set and several profiles are configured, the import is refused rather than restoring a session that cannot run: set a default with `meka profile use <name>`, or name one for the import with `meka --profile <name> session import`.

`updated_at` is stamped to the import time rather than restored from the export, so that restoring an archive older than a configured `retention` window isn't undone by the retention sweep on the next launch. `created_at` still carries the original.

## Forking a session

Branch off an existing conversation without disturbing it:

```bash
meka session fork 550e8400-e29b-41d4-a716-446655440000
```

The copy starts with the original's full conversation and continues from there under a new id, which is printed on stdout so it can be captured:

```bash
meka -r "$(meka session fork 550e8400-e29b-41d4-a716-446655440000)"
```

Use it to try a different direction from a known-good point, to run a throwaway question against a large accumulated context, or to keep a conversation you're about to compact.

What the copy carries: the full event log, scratchpad entries, working directory, permission level and approvals switch, additional workspace roots, and cumulative stats. What it does **not**: sub-agent sessions (the sub-agent's result already sits in the parent conversation as a tool result, so the copy is complete without them), and the timestamps, which are stamped fresh.

A fork of an ordinary session records no link back to the one it came from; it is a root session
like any other. A fork of a *sub-agent* is the exception: it keeps that sub-agent's parent and
spawn terms, so the copy is a sibling under the same parent rather than a promotion to a session of
its own, and it is continued through `agent_followup` like any other sub-agent.

Forking copies what has been committed to the store, and only between turns. A session another meka process has open is refused rather than copied (see [Session locking](#session-locking)), and so is one with a turn in flight in the process that holds it: `POST /v1/sessions/{id}/fork` answers `409` `turn-in-flight` and ACP `session/fork` answers `InvalidParams`. The user message is persisted before the model is called, so a copy taken mid-turn would end on a prompt with no reply and restore as an unusable session. Cancel the turn or wait for it.

The same operation is available from the REPL as `/fork`, which switches you into the copy and leaves the original where you branched (an `always` or `never` given at an approval prompt stays with the original; see [Permissions](permissions.md#approvals)); over HTTP as `POST /v1/sessions/{id}/fork`; and over ACP as `session/fork`.

### Fork or export/import?

Both produce a runnable copy under a new id. Reach for `fork` to branch a conversation you're working on, and for `export` + `import` to move a session between machines or keep an archive. Export/import also copies sub-agent transcripts and preserves `created_at`, because an archive should restore whole.

## Rewinding a session

Drop the most recent turns from a session:

```bash
meka session rewind 550e8400-e29b-41d4-a716-446655440000
meka session rewind 550e8400-e29b-41d4-a716-446655440000 -n 3
```

The cut lands on a turn boundary, so a tool call is never separated from its result, and nothing is deleted: the dropped turns stay in the event log and still appear in `meka session export`, marked at the point of the rewind. The model simply stops seeing them.

The command takes the session lock, so it refuses to run while a REPL, `meka serve`, or `meka acp` holds the session; that process has its own copy of the conversation in memory and would write over the rewind on its next turn. In the REPL use `/rewind` instead. Under ACP or the HTTP API there is no in-session equivalent, so close the session in the editor (or stop the server) and run this command.

Its main use is recovering a session a provider has started rejecting. A provider validates the whole conversation on every request, so one piece of content it rejects fails every later turn too.

meka repairs a rejection caused by content added since the last request the provider accepted, and repairs a mislabeled image on resume, but anything older than that needs rewinding past. That window is usually the current turn, and it reaches back into the previous one when a turn failed mid-tool-loop and left it unaccepted. A compaction widens it to the whole conversation, because a compaction replaces that conversation wholesale and nothing in the result has been accepted yet. The repair escalates: first it removes the attachments the turn added and leaves everything else alone, and only if that is refused as well does it empty the turn's tool calls, moving each call's arguments into the result that reports it and replacing the result's body with an explanation. The second step exists because a tool result is usually text, which the first step cannot touch, and because a call's own arguments can be what the provider objected to. A step the provider then accepts is not counted as spent, so the cheap one stays available if the turn is refused again later.

Neither step changes the shape of the conversation: a tool call stays a tool call and its result stays its result, marked as an error. That is deliberate. Removing one half of a pair is the one thing every provider refuses outright, so a repair that could do it might turn a recoverable rejection into a permanent one. The model sees a failed tool call, which it already knows how to read, with the arguments it sent quoted in the failure so it can tell which call not to repeat.

Nothing a repair removes is deleted. The log is append-only, so the superseded messages stay on disk and [`meka session export`](#exporting-a-session) renders them above a marker saying what replaced them. Use `--format json` to get a removed attachment back: the markdown export writes each message as its text and leaves image blocks out. Only the conversation the model sees is changed.

Whatever is removed is restored untouched if the retry carrying it is refused too. That restore is what bounds the risk, and it bounds it only in that direction: each step spends a fresh retry sequence rather than a single request, and a step whose retry *succeeds* keeps the loss, so the trigger is deliberately narrow. The words you typed are never rewritten by either step, though an image you attached to that prompt is exactly what the first one removes, replacing it with a note. Notes meka inserts into a conversation are prefixed `[meka harness]`.

Both steps run whether the provider answered `400` or spent every retry on a `5xx`. A gateway in front of a model reports a payload its own decoder choked on as a server error, which is indistinguishable from being overloaded, so meka honors the retries in full and treats a refusal that outlives them as one the content may explain.

On the `5xx` path it does one thing more before touching anything. The retry sequence is short by design (two attempts across three seconds of backoff), which an ordinary overload outlasts, so meka waits eight seconds and sends the *same* request one last time. If that succeeds the outage was the whole story and nothing is lost; if it fails too, the reading that the body is the problem has been earned rather than assumed. The wait is paid only by a turn that was otherwise about to start deleting things, and only once per run of consecutive failures: a request the provider accepts makes it available again, since a later refusal is about work the earlier wait never saw. A `400` skips it, because the provider has already read the body and said no.

Nothing outside those two shapes degrades at all, because a degraded retry that succeeds only because the network came back would keep the loss. Excluded, then: a dropped connection, which never delivered the request for anything to judge; a `429`, which is a statement about rate rather than about what was sent; a failure that arrives partway through a stream, after some of the answer has reached you, both because re-sending would print it twice and because the stream cannot tell an overload from anything else; and anything at all once the retries have *not* been exhausted. If a step it did try does not help, it says so and points here before the turn fails.

One cause of that refusal has its own fix. A session recorded by 0.41 can hold a `tool_result` whose content is a bare JSON string, a shape meka does not read: the row is dropped as the session loads, which leaves the `tool_use` it answered unanswered, and the provider rejects the next turn over the mismatch. Run the [one-shot upgrade script](../getting-started/upgrading.md), which converts those rows in place, rather than rewinding past a turn you wanted to keep.

## Deleting sessions

Delete specific sessions by id:

```bash
meka session delete 550e8400-e29b-41d4-a716-446655440000
```

Delete multiple sessions at once:

```bash
meka session delete 550e8400-e29b-41d4-a716-446655440000 a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

Delete every session not updated in the last N days:

```bash
meka session delete --older-than-days 90
```

This is the manual counterpart to [`retention`](#history-retention). It can't be combined with ids or `--all`, and `0` is refused: it would match everything.

Delete all sessions:

```bash
meka session delete --all
```

`--all` takes no ids of its own: naming some sessions and then asking for every session are two
different requests, and it refuses rather than quietly doing the wider one.

## Input history

Separate from your saved conversations, meka keeps a rolling history of the prompts you *type* at the REPL, so **Up-arrow** recall and **Ctrl+R** reverse-search work across runs (shell-style). This is distinct from a session (a stored conversation) and from the `/history` slash command (which reprints the current conversation).

List recent input-history entries (oldest first; `-n 0` shows all), one per line, or as
`{"history": [...]}` with `--format json`:

```bash
meka history list
meka history list -n 100
meka history list --format json
```

Clear it entirely:

```bash
meka history clear
```

## Managing sessions via SQLite

You can also manage sessions directly through the store's SQLite file. For example, to list all sessions:

```bash
sqlite3 ~/.local/share/meka/meka.db \
  "SELECT id, created_at, updated_at FROM sessions ORDER BY updated_at DESC;"
```
