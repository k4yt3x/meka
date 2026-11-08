# HTTP API

`meka serve` exposes meka as an HTTP API server so other programs can drive agent turns programmatically. Where [Interactive mode](./interactive-mode.md) is for humans at a terminal and [ACP](./acp.md) is for editor integrations over stdio, the HTTP API is for **service-to-service** use cases:

- A Telegram or Discord bridge that connects a chat bot to an agent.
- A web or mobile UI that streams assistant responses in real time.
- A script or orchestrator that embeds meka as a sub-agent backend.
- Any cross-language client that speaks HTTP+JSON.

All three entry points (`meka`, `meka acp`, `meka serve`) drive the same agent core: same tools, same profiles, same session persistence. The HTTP API is a transport layer on top. A shell script that wants one turn as JSON and no server can use `meka --oneshot --format json` instead; see [One-shot mode](./one-shot-mode.md#json-output).

## Starting the server

```bash
meka serve
```

The server reads the `[serve]` section from your `config.toml` (see [Configuration](#configuration) below). At minimum you need one bearer token; `bind` defaults to `127.0.0.1:8080`:

```toml
[serve]
bind = "127.0.0.1:8080"

[[serve.tokens]]
token = "${MEKA_API_TOKEN}"
scopes = ["sessions:r", "sessions:w"]
```

On startup the server logs the bind address and begins accepting requests. All endpoints (except health probes and OpenAPI docs) require a valid `Authorization: Bearer <token>` header.

The server starts whether or not a profile is configured, unlike every other way of running meka, so an installation managed over the API can come up before its first account and profile exist and have them added through [`POST /v1/accounts` and `POST /v1/profiles`](#managing-the-installation). Until a session can get a profile, `GET /v1/health/ready` answers `503` with `profile_configured: false`, and a `POST /v1/sessions` that names no profile is a `422` carrying the configuration's own sentence: no profile configured, or several and no `default_profile`. A body that names a configured `profile` is served either way.

Two flags are refused rather than ignored: `-c` and `-r`. Both name one run's session, and the server creates one per `POST /v1/sessions`, each naming its own profile. Address an existing session by id under `/v1/sessions/{id}`, and pass `profile` on the create request. `--profile` is accepted, since it selects which configured profile a session gets when it names none, which is a property of the server rather than of one session.

> **TLS**: `meka serve` speaks plain HTTP. For production, front it with a TLS-terminating reverse proxy (nginx, Caddy, Cloudflare Tunnel).

Under systemd, the unit lines in the shell tool's [Memory](../tools/shell.md#memory) section keep a command that takes every byte from taking the server with it.

## Quick example

### Blocking turn (simplest)

```bash
# Create a session
curl -s -X POST http://localhost:8080/v1/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"working_directory": "/home/user/project"}' | jq .id
# → "550e8400-e29b-41d4-a716-446655440000"

# Submit a turn
curl -s -X POST http://localhost:8080/v1/sessions/550e8400-e29b-41d4-a716-446655440000/turns \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "list the files in src/"}' \
  | jq -r '[.messages[] | select(.role == "assistant") | .content[] | select(.type == "text") | .text] | join("")'
# → "Here are the files in src/: ..."
```

### Streaming turn

```bash
curl -N -X POST http://localhost:8080/v1/sessions/$SESSION_ID/turns \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "explain this codebase", "stream": true}'
```

The response is a `text/event-stream` (SSE) that emits typed events as the agent works:

```
retry: 3000

event: turn.started
id: 1793001600000000
data: {"turn_id":"...","session_id":"...","started_at":"2026-05-26T13:45:12Z","source":"client"}

event: assistant_text.delta
id: 1793001600000001
data: {"text":"This project is ","turn_id":"...","session_id":"..."}

event: assistant_text.delta
id: 1793001600000002
data: {"text":"a Rust workspace that..."}

event: tool_call.composing
id: 1793001600000003
data: {"id":"tu_1","name":"file_read"}

event: tool_call.executing
id: 1793001600000004
data: {"id":"tu_1","name":"file_read","input":{"path":"src/main.rs"},"display_summary":"src/main.rs"}

event: tool_call.completed
id: 1793001600000005
data: {"id":"tu_1","is_error":false,"content":[{"type":"text","text":"fn main() { ... }"}]}

event: turn.finished
id: 1793001600000012
data: {"turn_id":"...","session_id":"...","stop_reason":"end_turn","usage":{"input_tokens":12340,"output_tokens":567,...},"position":4127}
```

Every payload carries `turn_id` and `session_id`; the later events above elide them. An `id` is an integer that rises by one per event across the session's turns. Each time a server loads the session, after an eviction or a restart, the sequence starts again above every id it issued before, at the microseconds since the epoch, so a `Last-Event-ID` from before always reads as behind it.

### Detached turn

```bash
curl -s -X POST http://localhost:8080/v1/sessions/$SESSION_ID/turns \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "refactor the parser", "detach": true}'
# → 202 Accepted, Location: /v1/sessions/.../turns/...
# → {"id":"...","source":"client","started_at":"2026-10-17T09:12:03Z","permission":"workspace","usage":{...}}
```

The turn runs whether or not this client stays. Its events are on the [session feed](#the-session-feed), and its ending on the resource `Location` names; see [Holding a turn](#holding-a-turn).

## Core concepts

### Sessions

A session is a persistent conversation with its own working directory, permission level, and message history. Sessions live in the same store as screen and ACP sessions; they're interchangeable.

```
POST   /v1/sessions           Create a session
GET    /v1/sessions           List sessions (paginated), or with ?q= find them by their words
GET    /v1/sessions/{id}      Get session details
PATCH  /v1/sessions/{id}      Update permission, approvals, working_directory, profile, title or pin
DELETE /v1/sessions/{id}      Delete the session
POST   /v1/sessions/{id}/fork Branch a copy off a session
```

`GET /v1/sessions` is paginated. `limit` is how many to return (default 50, clamped to 1..200).
Pinned sessions come first, newest pin on top, then the rest most recently updated first. When
more remain, the response carries `next_cursor`; pass it back as `cursor` for the next page.
`include_children=true` lists sub-agent sessions too, `working_directory=<path>` keeps only the sessions in
that working directory, compared in the canonical spelling every session records, and
`parent=<id>` lists one session's direct children: the sub-agents it spawned, which is what a
tree view asks for per expanded node, and what an audit asks to learn what a session dispatched.
`parent` names sub-agents by definition, so `include_children` is moot beside it; a session with
none, or an id no session has, is an empty page. `profile=<name>` keeps the sessions on one
profile, `project=<name>` those on one [project](projects.md) (`project=` those on none),
`pinned=true` or `pinned=false` the pinned or the unpinned ones, and
`updated_since=<RFC 3339>` the ones whose `updated_at` is after that instant: a turn or a change
to what the session runs as, never a title or a pin. The filters compose, and apply to a
[search](#titles-pins-and-search) too.

When creating a session, specify the working directory and optionally a permission level, the
approvals switch, a profile, a project, and capabilities:

```json
{
  "working_directory": "/home/user/project",
  "permission": "workspace",
  "approvals": false,
  "profile": "work",
  "project": "webdev",
  "capabilities": {
    "supports_reasoning_stream": false,
    "supports_permission_prompts": true
  }
}
```

`approvals` is whether a tool call above the level is put to the client for approval rather than
refused; see [Approvals](#approvals) below. Omitted, it is the server's `[permissions].approvals`.
Every session response echoes both `permission` and `approvals` back.

Create, get, list, fork and `PATCH` all answer with the same session record:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-05-26T13:45:12Z",
  "updated_at": "2026-05-26T13:47:01Z",
  "working_directory": "/home/user/project",
  "permission": "workspace",
  "approvals": false,
  "profile": "work",
  "project": "webdev",
  "title": "list the files in src/",
  "last_turn": {
    "id": "7f3b...", "source": "client", "started_at": "2026-05-26T13:46:40Z",
    "permission": "workspace", "ended_at": "2026-05-26T13:47:01Z", "status": "succeeded",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12340, "output_tokens": 567, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 8000}
  },
  "turn_count": 4,
  "usage": {"input_tokens": 48211, "output_tokens": 2304, "cache_creation_input_tokens": 9120, "cache_read_input_tokens": 31800},
  "capabilities": {"supports_reasoning_stream": false, "supports_permission_prompts": true},
  "turn_in_flight": false,
  "requests_pending": 0,
  "checklist": [
    {"id": 3, "text": "write the tests", "status": "in_progress"},
    {"id": 4, "text": "ask Sam about the venue", "status": "deferred", "reason": "waiting on Sam"}
  ]
}
```

`checklist` is the session's open [checklist](checklist.md): every item the agent has committed to
and not yet disposed of, with its `id`, `text`, `status` (`pending`, `in_progress` or `deferred`),
and on a deferred item its `reason` and the `task` it waits on when it waits on one. Present on a
session this server holds; the feed's `checklist.updated` carries the same list on every change.

`created_at` is when the row was made; `updated_at` moves on a turn and on any change to what the
session runs as (`permission`, `approvals`, `working_directory`, `profile`, `project`), a `PATCH`
included, but not on a title or a pin; `title` is the title a client set, else the first user
message's words, whitespace collapsed and cut to 200 characters. A pinned session also carries
`pinned_at`. `turn_count` is how many turns the session has on record, every host's, the turns [`GET
/v1/sessions/{id}/turns`](#turns) lists, and `usage` what they spent, in the four counters a turn's
`usage` has. A turn the [`turn_retention`](../configuration/config-file.md#sessionturn_retention)
sweep deletes leaves both. A sub-agent's turns are on its own record, not its parent's.

`last_turn` is the session's latest turn, whatever host ran it: the same record [`GET
/v1/sessions/{id}/turns`](#turns) lists, carried here because a list of sessions is rendered from
it, one row per session. `id` is the turn's id, the one `turn.started` announced it under and every
message it added names; `source` is who opened it (`client`, `inbox`, `schedule`, `background`,
`compaction` for a compaction run between turns, or `parent` for a sub-agent's turn); `started_at`
is when; `permission` is the level it ran under,
the one its opening told the model, and for a turn still opening, such as the record a detached
request's `202` carries, the session's level, which it opens under unless the level moves first.
Once the turn has ended, `ended_at` says
when and `status` how: `succeeded`, with its `stop_reason` (`end_turn`, `max_tokens` or `refusal`),
`failed`, with an `error` carrying the same `type` and `detail` a Problem Detail would (never the
provider's own text), or `canceled`; `profile` names the fallback that served the turn when the
session's own profile could not ([Fallback profiles](sessions.md#fallback-profiles)), and is absent
otherwise; and `usage` is what it spent: every round of it, and any compaction it ran. A turn
that began
and shows no `ended_at` is either in flight, which `turn_in_flight` says, or one the process died
under. Whether a turn is running now is a live fact of the process that no row stores. Together they
answer the one question a session list asks: running, waiting on a person (`requests_pending` above
zero), or stopped and how. A turn refused before it began, by a `4xx` or the `503` for a required
MCP server, added nothing and leaves `last_turn` alone. A compaction run between turns, by
`POST /compact` or the screen's `/compact`, is a turn of its own with `source: "compaction"`, under a
`turn_id` equal to the compaction's `compaction_id`: its row says how it ended and what it spent,
and the compaction's history item names it.

On every response and event this API sends, a field that has no value is omitted rather than sent
as `null`. On a session that means `last_turn` until a turn has begun, `requests_pending` on a
session this server does not hold, `working_directory` when the row recorded none, `permission` when the session
is not loaded and its row records no level, and `parent_id`, which only a sub-agent's session
carries. The same rule gives a turn its `refusal_text` only on a refusal and a tool call its
`display_summary` only when the tool has a label. It is also why `meka session show --format json`
prints the same object for the same record: the row's fields are one shape shared by both
surfaces, and `capabilities`, `turn_in_flight`, `inbox_pending`, `requests_pending` and `checklist`
are what the server adds to it.

`profile` names a profile in the server's `config.toml`; `GET /v1/profiles` lists them, and a name
that is not configured is a `422` whose `detail` reads `no profile named 'x' (configured: a, b)`.
Omitted, it is the server's own default profile. The session keeps
it for the rest of its life and every session response echoes it back as `profile`, so a client can
confirm which account a session bills.

`project` names a [project](projects.md), which decides the memory tier the session's writes land
in and the tiers it sees; `GET /v1/projects` lists them, `POST /v1/projects` adds one, and a name no
project has is a `422` whose `detail` reads `no project named 'x' (configured: a, b)`.
Omitted or empty, the session is on no project, and every session response echoes `project` back,
`""` for none. `PATCH /v1/sessions/{id}` with `{"project": "movie"}` moves a session onto a project
and `{"project": ""}` takes it off one; the row is rewritten, and the next turn's context restates
the memories the session now sees. Like `profile`, it is a `409` when a turn is in flight.

To move a live session onto another profile, `PATCH /v1/sessions/{id}` with `{"profile": "other"}`.
That rewrites the session's row, so it holds for a resume from any surface rather than for this
request. Switching mid-conversation is allowed and is your call: a thinking block is tagged with the
backend that produced it and is not replayed to a different one, so from the next turn the model no
longer sees the reasoning recorded under the old profile. Like `working_directory`, it is a `409` when a turn is
already in flight; cancel first. A `PATCH` naming any of `working_directory`, `profile` and
`project` holds the session while it writes, the way compact does, so a turn arriving meanwhile
answers `409` rather than starting under it, and so does a second such `PATCH`. `permission` and `approvals` are the two fields that apply during
a turn; see [Permission levels over HTTP](#permission-levels-over-http).

A `PATCH` naming a profile moves the session to that profile, and the profile is the whole story:
the model, the account and every model-tied setting come from it, so there is nothing else on the
row to reconcile. The row is also the billing record, so a profile the store cannot record fails the
request; `permission`, `approvals` and `working_directory` still apply to the live session when their row write
fails, and the failure is logged.

If you run more than one `meka` on the same store, send the `PATCH` to whichever process has the
session. A body naming only a profile is the one `PATCH` that works on a session this server has
not loaded, and it takes the session lock to do it, so a session another process is running answers
`409` `session-locked` rather than moving a row that process would go on ignoring. Only the host
holding a session may change what it runs on.

A body naming **only** `profile` is also the rescue for a session whose profile has left
`config.toml`: it moves the row without building an agent, so it works on a session that cannot
currently run. Adding `permission` or `working_directory` to the same body loses that, because those need a loaded
session and loading one is exactly what fails; send the profile on its own first.

The `working_directory` field is validated on create, fork and patch, by the same rule the screen's `/cd` and ACP
apply:

- Must be an **absolute path** (no relative paths).
- Must **exist** on the server's filesystem.
- Must be a **directory** (not a file, device, or socket).
- Must not contain **null bytes** (which cause kernel/userspace path mismatch).
- Is recorded, and echoed back, in its **canonical spelling**: symlinks resolved, `.` and `..`
  removed. Every host records the same spelling, so a `working_directory` filter on the listing finds a session
  however its directory was spelled.

If `working_directory` is omitted, it defaults to the server process's current working directory.

Sessions persist server-side until explicitly deleted or evicted by the idle timeout GC (see [Session lifecycle](#session-lifecycle)).

#### Capabilities

| Capability | Default | Meaning |
|------------|---------|---------|
| `supports_reasoning_stream` | `false` | Include `thinking.delta` events in the SSE stream |
| `supports_permission_prompts` | `true` | The client can answer a mid-turn `approval.requested` event |
| `answers_prompts_on_return` | `false` | A prompt in a turn no request holds waits for this session's client to come back and answer it; an elicitation waits an approval's thirty minutes there, or as long as the call that asked, if that is sooner |
| `supports_elicitation` | `false` | An MCP server's elicitation parks as an `elicitation.requested` event for the client to answer; see [Elicitations](#elicitations) |

Enabling `supports_reasoning_stream` costs a *streaming* turn its retry on a transient provider failure: the deltas have already reached you and a second attempt would repeat them, and reasoning is the first thing a turn produces. Blocking turns on the same session are unaffected, since they carry whole blocks rather than deltas.

Set `supports_permission_prompts: false` if you stream but have no interface to show an approval
prompt on, which is the normal case for a service-to-service client streaming for liveness. Gated
tools are then denied immediately with an explanatory `notice`, the same as blocking mode. Leaving it
`true` means every gated call parks for 30 minutes and then denies anyway, which is hard to tell
apart from a hang. Better still, create the session with `permission: "workspace"` so nothing is gated.
The flag speaks for the streaming client: a feed reader that opened the stream with `attend=true`
is asked regardless, since attending is that declaration made per connection.

Set `answers_prompts_on_return: true` on a session a person drives through a client that comes and
goes, a browser tab say. A gated call in a turn no request holds, a detached turn, an inbox turn, a
scheduled fire, then parks whether or not anyone is attending, survives the last attending reader
leaving, and closes only on an answer, the turn's cancellation or its 30-minute expiry; the client
that comes back is handed it with the feed's replay and answers it. The session is busy for as
long as the prompt waits. A held turn keeps the grace rule, since its request leaving stops the
turn anyway. Leave it `false` for a script or a service with no interface to show a prompt on,
where the default refusal is the right answer.

#### Forking a session

`POST /v1/sessions/{id}/fork` copies a session's conversation into a new session and returns it with
`201` and the usual session body. The copy starts with the source's full history and is immediately
usable; the source is left untouched, and does not have to be in memory, so a GC-evicted session
forks as well as a live one. A source with a turn in flight answers `409` `turn-in-flight`, as
`PATCH`, `DELETE`, compact and rewind do; cancel the turn or wait for it. A source another meka
process holds answers `409` `session-locked`. Either copy would have ended on a prompt nothing
answered.

The body is optional and inherits everything by default. It takes two fields. `working_directory`
moves the copy, matching ACP's `session/fork`, which likewise carries a workspace but no permission
or capability fields. `turn_id` branches before a turn: the copy holds the conversation as it stood
before that turn, as a rewind of the copy to it would leave it, so the copy's history ends with that
rewind, and the source keeps everything. A turn the source's context does not hold (another
session's, one a compaction summarized, a compaction's own, or one that left nothing in it) is a
`422` naming it. An `If-Match` header holding the conversation's `ETag`, from `GET
/v1/sessions/{id}/history` or `/context/messages`, makes the copy conditional on the source being as
it was read; see [Editing the conversation](#editing-the-conversation).

```json
{ "working_directory": "/home/user/other-project", "turn_id": "7f3b..." }
```

Branching is this one call. A fork gives the turns it copies ids of their own, so a turn id of the
source names nothing on the copy, and a rewind of the copy cannot be aimed by it.

Permission, approvals, capabilities and the profile are inherited and remain changeable afterwards via
`PATCH /v1/sessions/{id}`. Sub-agent child transcripts are not copied, and a fork of an ordinary
session records no link back to its source.

A sub-agent's own id is refused with `422`: the copy would keep that sub-agent's parent and spawn terms,
so it is a sibling under the same parent rather than a session this endpoint could hand back. See
[Forking a session](./sessions.md#forking-a-session) for the full semantics.

#### Sub-agent sessions cannot be driven through this API

`GET /v1/sessions?include_children=true` lists the sessions an `agent_spawn` created. Those ids are
readable through every endpoint on this page (`/history`, `/context`, `/context/messages`, `/turns`,
`/export`) and drivable through none of them: `POST /v1/sessions/{id}/turns` answers `422` with
`/errors/session-not-drivable`, as do `/compact`, `/responses/{request_id}`, `/fork`, a
`POST /v1/jobs` naming one, and `PATCH /v1/sessions/{id}`. A sub-agent
runs under the tools, permission ceiling and profile its spawn call set, which live in its
spawn record and which only its parent can reconstruct, so the conversation is continued with the
`agent_followup` tool from the parent rather than over HTTP.

A sub-agent is never a session this server holds: its parent runs it under the parent's runtime.
Its record still says when it is in a turn, because the parent's run registers it for exactly as
long as it runs, so `turn_in_flight` on a sub-agent's record and in a listing is true while a
spawn or a follow-up is running it; `last_turn` on its row says how its last run ended. While it
runs, its id names its own event feed, which is read rather than attended; see
[A sub-agent's feed](#a-sub-agents-feed). A prompt a sub-agent's call parks rides the parent's
feed, with the sub-agent named, and the sub-agent's feed mirrors it; see [Approvals](#approvals).

Two exceptions, both of which change a transcript without running anything on it. Teardown stays
open: `DELETE /v1/sessions/{id}` discards a sub-agent and `DELETE /v1/tasks/{task_id}` stops one of
its background tasks, and the parent's own `agent_delete` does the same thing. So does
`POST /v1/sessions/{id}/rewind`, which takes turns out of a context the same caller can already read
in full through `/history`, and which `meka session rewind` has always allowed on a sub-agent. The
line is whether the model runs: `/compact` is refused because compaction is a turn.

#### Importing an archive

`POST /v1/sessions/import` recreates a session tree from a `meka session export` archive under fresh
ids, on the same terms as the CLI's `meka session import`. An archive naming no profile takes the
server's default, the same one `POST /v1/sessions` applies to a body with no `profile`; a
long-lived host always has one, since it refuses to start without it.

One limit is the server's alone: an archive holding more than **1000** sessions is refused with a
`422` whose detail names the count and the cap, and points at `meka session import`. The whole tree
is written in one transaction on the process's single connection to the store, so a larger one would
hold every other in-flight request behind it. A one-shot `meka session import` restoring its own
backup has nothing to contend with and so carries no cap; it is the way to restore a tree this
large.

Everything else about the archive is honored as the CLI honors it; see
[Exporting a session](./sessions.md#exporting-a-session).

#### Titles, pins and search

A session is labeled by its first words until someone titles it. `PATCH /v1/sessions/{id}` with
`{"title": "Research notes"}` sets the title every surface shows from then on, the CLI and editor
clients included, and answers with the updated record; an empty string clears it. A title is one
line of at most 200 characters, whitespace runs collapsed, and a longer one is a `422`. `{"pinned":
true}` pins the session, which lists it first on every surface and spares it from the retention
sweep; `{"pinned": false}` unpins it. Pinned sessions order among themselves by `pinned_at`,
newest first, and pinning an already pinned session keeps its time. Neither field moves
`updated_at`, so a rename does not reorder the list, and a body naming only these two, or only
`profile`, works on a session this server has not loaded and on one another process is running.

`GET /v1/sessions?q=<words>` searches: it answers `{"sessions": [...]}` with the sessions whose
conversations hold the words, best first, each the same record the listing carries plus an
`excerpt`, the line of the best-matching message the words were found on, whitespace collapsed and
cut short, marked `(summary)` when taken from a compaction summary, and `excerpt_position`, the
position of that message: `GET /v1/sessions/{id}/history?before=<excerpt_position + 1>` shows it at
the end of a page, since `before` and `after` leave out the position they name. What is searched is
what was said, user messages, replies and compaction summaries; tool calls, tool results and
thinking are not, and neither is a message a rewind took back, which the history no longer shows. A
session whose title holds every word comes first, with an `excerpt` only when its words matched too;
the rest are ranked by their best message the history shows. Every word must appear; when nothing
holds them all, the words are tried as prefixes, then sessions holding any of them, then any as a
prefix. Words match by their stem and with every diacritic folded; scripts written without spaces
match character by character. Every listing filter applies beside `q` (`include_children`,
`working_directory`, `parent`, `profile`, `project`, `pinned`, `updated_since`). `limit` is how many
at most (default 20, clamped to 1..100): a search answers its best `limit` and carries no
`next_cursor`, so a `cursor` beside `q` is a `422`. A blank `q` is no `q`, and answers the listing.
Requires `sessions:r`.

#### Detecting an in-flight turn

Session responses carry `turn_in_flight`, a boolean saying whether a turn is running right now. It
exists so a client whose connection dropped can tell "my turn is still running" from "my turn died"
without submitting a speculative turn and reading the `409`. A dropped connection does not cancel
the turn at once: a held turn runs until its next round boundary after the grace, a detached one to
its end, and resubmitting would duplicate a reply the user is about to receive. Poll
`GET /v1/sessions/{id}` and wait for it to go `false` rather than retrying blind.

The same holds for a held JSON turn whose client gives up: a request timeout on your side does not
stop the turn until the grace is out, and a turn with a single round that is already under way ends
as it would have. Read what completed from `GET /v1/sessions/{id}/history`, and the turn's ending
from `GET /v1/sessions/{id}/turns/{turn_id}`. A client whose timeout is shorter than its longest
turn sends `detach: true` instead; see [Holding a turn](#holding-a-turn).

`turn_in_flight` answers for this process: a session it holds, and a sub-agent it is running under
a parent. A session another process is running, a screen open on it say, reads as idle, because only
that session's file lock knows and a read never probes the lock. Once a turn has ended, however it
ended, `last_turn` on the record says how.

### Turns

A turn is one round-trip: you send a user message, the agent processes it (potentially calling tools in a loop), and returns a result. Every turn that began is a row: `GET /v1/sessions/{id}/turns` lists them, newest first, each with its `id`, `source`, when it began and ended, how it ended and what it spent, in the shape `last_turn` on the session record shows for the latest one. `limit` is how many at most (default 50, clamped to 1..200), and when more remain the response carries `next_before`, the id to pass back as `before` for the page of turns older than it. A [history](#history) item names the turn that said it in `turn_id`, which is also the id that turn's `turn.started` announced.

```
POST   /v1/sessions/{id}/turns             Submit a turn
POST   /v1/sessions/{id}/cancel            Cancel an in-flight turn
GET    /v1/sessions/{id}/turns             The turns that began, newest first
GET    /v1/sessions/{id}/turns/{turn_id}   One turn, with its ending once it has one
```

**One turn at a time per session.** A second `POST /turns` while another is running returns `409 Conflict`. Across sessions, turns run fully concurrently. A client that would rather hand the message over and be told when the model read it uses the [inbox](#the-inbox) instead of waiting for the session to be free.

`POST /cancel` takes an optional body `{"turn_id": "..."}`. Without one it stops whatever is running. With one it stops only that turn, and answers `409` `turn-mismatch` naming the turn actually in flight when it is another, so a client that watched a turn cannot stop the scheduled fire or the inbox turn that replaced it. Every `turn.started` on the [session feed](#the-session-feed) carries the id to name.

The turn request body accepts six fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message` | string | *(required)* | The user message. May be empty when `images` is non-empty |
| `images` | array | `[]` | Image attachments; see [Image attachments](#image-attachments) |
| `stream` | bool | `false` | How the response is encoded: `false` → JSON; `true` → SSE stream |
| `detach` | bool | `false` | Whether this request holds the turn; see [Holding a turn](#holding-a-turn) |
| `options.skill` | string \| null | `null` | When set, activates the named [skill](./skills.md) for this turn (equivalent to `/skill <name>` on the screen). With an empty `message` the skill body runs alone, as `--skill` does; a turn with no text, no image and no skill is a `422`, as is a skill the session's [project](./projects.md) hides |
| `options.unanswered_message` | string | `keep` | What becomes of `message` if the turn ends, failed or canceled, before anything from the model reached the conversation. `keep` leaves it in place, as the screen does with a typed prompt. `withdraw` takes it back, for a client that resends a failed turn; see [Resending a failed turn](#resending-a-failed-turn) |

### Holding a turn

`stream` and `detach` are independent. `stream` says how the response is encoded. `detach` says whether the request holds the turn: a held turn's response ends with the turn, and the turn is stopped at its next round boundary once nobody has been listening for [`[serve] stream_reattach_grace`](../configuration/config-file.md#servestream_reattach_grace), 30 seconds by default. A detached turn's request is only the trigger: the turn runs for the session like an inbox turn, and only `POST /cancel`, a shutdown or its own failure ends it early.

| | `detach: false` (held) | `detach: true` |
|---|---|---|
| `stream: false` | `200` with the turn record and what it wrote once it ends; see [Blocking response](#blocking-response). A dropped request stops the turn after the grace. | `202 Accepted` at admission. The body is the turn record and `Location` names it under `/turns/{turn_id}`. |
| `stream: true` | SSE through the terminal. A dropped stream stops the turn after the grace. | SSE from admission. Drop it any time; the turn runs on. |

A held request is for a client that is there for the turn: a killed `curl` should stop spending tokens whichever encoding it used. A detached request is for a client that cannot hold a connection open or does not want to, a browser tab that may close, a job runner behind a proxy with a short read timeout. The turn record a detached request is answered with is the row `GET /v1/sessions/{id}/turns` lists, with no ending yet; read `GET /v1/sessions/{id}/turns/{turn_id}` for the ending, or take it from the feed's terminal event. The feed may announce `turn.started` before the 202 reaches you; correlate by `turn_id`.

A listener is any reader of the session feed, so a held turn with a feed subscriber attached is not stopped for its own request leaving. A detached stream is a feed reader narrowed to one turn: dropping it has no effect on the turn, and a reader that fell behind is caught up from the ring rather than ending the turn, as a feed reader is.

### The inbox

`POST /turns` runs one turn for one request, the response scoped to it, and answers `409` while the session is busy. The inbox is the asynchronous door beside it, for a client that lives with a session rather than driving one turn at a time: a chat bridge, a UI, a parent process with something to say while the agent works.

```
POST   /v1/sessions/{id}/inbox              Enqueue a message
GET    /v1/sessions/{id}/inbox              Items the model has not been shown
DELETE /v1/sessions/{id}/inbox/{item_id}    Withdraw an item still waiting
```

The body is `{"message": "...", "class": "steer" | "followup" | "interrupt", "source": "..."}`. `class` is required and is the whole contract:

- **`steer`** reaches a turn that is already running. The loop reads the inbox at every round boundary, after a round's tool results and before the next request, and appends what it finds to that same message, so the model sees it as soon as it next asks the provider anything. The end of a reply is a boundary too: a steer that lands while the model writes the answer that would end the turn follows that answer, and the turn goes on to answer it. This is what lets you correct or redirect the agent ten seconds into a ten-minute task, the way a person glances at a message mid-task.
- **`followup`** waits for the running turn to end.
- **`interrupt`** does not wait for the answer being written. While the provider is streaming, the stream is dropped within a second, the text that had arrived is kept as the answer so far, and the message follows it as the next thing the model reads; the same turn carries on, with no terminal in between, and the feed says so with a `notice`. A cut can land inside a thinking block or after a tool call was announced and never run: the `notice` is what closes those. While a tool runs, nothing is cut: the message is read at the round boundary after the tool's results, exactly as a `steer` is. A profile that does not stream has nothing partial to keep: the reply being generated is dropped whole and the request goes again with the message. The cost is the request sent again, which the prompt cache mostly absorbs, and the part of the answer that was never written. For "stop, do this instead"; a `steer` is enough for "also, when you get to it".

Every class rides the opening of the next turn when nothing is running, whoever starts that turn, and opens a turn of its own when nothing else does. A turn that has been admitted but has not yet sent its first request is not running yet in this sense: an item that lands in that moment rides its opening, whatever its class. The model reads each item under a header meka writes: `[Message from <source>, arrived <time>]`, with `while you were working` added when it landed mid-turn. Without a `source`, or with a blank one, the header is `[Message arrived <time>]` and names nobody: who sent a message is the client's to say, and meka does not guess it from the token that carried it. `GET /v1/sessions/{id}/inbox` omits `source` for such an item. The body is verbatim, so a client relaying text from strangers fences it itself.

The answer is `202` with the item and its state:

```json
{"item_id": "...", "session_id": "...", "class": "steer", "state": "pending", "replayed": false,
 "retry_until": "2026-10-18T15:02:11Z"}
```

`retry_until` is the deadline for retries, the item's enqueue plus [`[serve] inbox_retry_ceiling`](../configuration/config-file.md#serveinbox_retry_ceiling): an attempt that fails after it is not retried, and the item is given up on. An item not yet tried still runs when its turn comes, whatever the time, so under a short ceiling a fresh item can show a `retry_until` already past and be delivered all the same. A fresh item answers `pending`, unless a turn already running read it in the meantime. A replayed key answers with what the earlier item has become: beside its `state`, the `not_before` and `last_error` it shows while it waits, or the `failure` and `last_error` it was given up on with, so a producer that lost its place learns the item's fate from the same call that enqueued it, without listing an inbox that no longer shows a withdrawn item.

An item is `pending` until its text is in the conversation, `appended` until the provider accepts a request carrying it, then `delivered`; `withdrawn` is a `DELETE`, a canceled turn, or meka giving up. **Delivered means the model read it**, not that it was written down: the feed's `inbox.delivered` fires when the provider accepts the request, and names the turn that carried the item. A rewind that takes an `appended` item's text out of the context before it was delivered offers the item again. `GET /v1/sessions/{id}` reports `inbox_pending` beside `turn_in_flight` while the session is loaded.

`Idempotency-Key` works here as on `POST /turns`: the same key with another `message` or `class` is refused with `409` `idempotency`. One difference matters to a bridge: the key is recorded on the row rather than in memory, so a retry that lands after a meka restart still finds its earlier item and answers it with `replayed: true`, and the key stays bound to that item for the session's life, withdrawn or delivered. The item is durable before the `202`; a session evicted for idleness is revived to run it, and a process that restarts finds it waiting.

A turn opened on inbox items that fails before anything from the model reached the conversation withdraws its prompt and offers the items again, when waiting can change the answer: the upstream could not serve the profile (an outage, a dropped stream, a quota or billing refusal), a required MCP server was not ready, or meka's own store or disk refused (a lock another process holds, a full disk). Any other failure, a rejected credential, a request the upstream judged wrong, a profile that cannot read the session's images, gives the item up at once, since every attempt would fail the same way. A retry waits 10 seconds, then twice that per attempt, up to five minutes between attempts, or less when the upstream said it would be back sooner (`Retry-After`, a quota reset); a stated time never makes it wait longer, since an account topped up early is noticed at the next attempt. Each deferral posts `inbox.deferred` on the feed and the webhook with the time of the next attempt and the error, and the item shows `not_before` and `last_error`, the same error as a Problem Detail (`type`, `title`, `status`, `detail`, never the upstream's text), while it waits. Retrying ends at `retry_until`, the item's enqueue plus [`[serve] inbox_retry_ceiling`](../configuration/config-file.md#serveinbox_retry_ceiling), an hour by default: the item is given up on, withdrawn, with `inbox.failed` on the feed and the webhook carrying the reason and the last error, so whoever is waiting is told rather than left with silence, and the item keeps that `failure` and `last_error` for a replayed key to answer; a canceled turn or a client's `DELETE` withdraws an item with no error on it, since none decided it; `"0s"` gives an item up on its first failure. An item a tool round already carried into the conversation is not retried: it is history, and the next turn of any kind delivers it. An item on a session another process holds (a screen open on it, say) is asked about again every ten seconds until the holder lets go, or the holder's own next turn carries it; those waits count toward the ceiling.

Only a `pending` item can be withdrawn. `DELETE` on one that is already in the conversation answers `409` `inbox-appended`, since only a turn can answer it now; a delivered or withdrawn one is `404`. A `DELETE` that lands in the instant between a turn reading the item and writing it may still be read by the model. Canceling a turn the inbox opened, with `POST /cancel`, withdraws the items it opened on, the way a canceled client turn loses its prompt, and the feed reports each as `inbox.withdrawn`. The endpoint refuses a sub-agent's session exactly as `POST /turns` does: a worker's inbox is its parent's, written with the `agent_steer` tool. No images in this release.

### Image attachments

Each entry in `images` is the bytes, `{"media_type": "...", "data": "<base64>"}`, or an image the
session's history already holds, `{"hash": "..."}`, the hash its image block carries on
`GET /history`. Bytes are inlined rather than referenced by path because the API is a network
surface: a client on another host shares no filesystem with the agent, so it can't name a file for
the agent to read. A hash lets a client send a message again, after an edit or a rewind, without
fetching and uploading the bytes: the stored bytes and media type are used, and only an image the
session holds can be named, the scope `GET /blobs/{hash}` serves. A `media_type` beside a `hash`,
or a hash the session holds no image for, is a `422` naming the entry.

```bash
curl -s -X POST http://localhost:8080/v1/sessions/$SESSION_ID/turns \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"what does this diagram show?\",
       \"images\": [{\"media_type\": \"image/png\", \"data\": \"$(base64 -w0 diagram.png)\"}]}"
```

- **Requires vision.** Attaching an image to a session whose profile has `vision = false` returns
  `422`. The check is per session, from the profile that session recorded, so a session created with
  `profile` or moved by a `PATCH` follows that profile rather than the server default. `vision` on
  each entry of [`GET /v1/profiles`](#discovery-endpoints) is the flag to read, for whichever
  profile the session runs on.
- **`media_type` is a hint.** If it doesn't name a supported format, the payload's magic bytes are
  used instead, so `application/octet-stream` still works for a real image.
- **Formats.** PNG, JPEG, GIF, and WebP pass through; BMP, TIFF, ICO, HDR, EXR, TGA, PNM, QOI,
  DDS, and Farbfeld are converted to PNG. Anything else is a `422`.
- **Size.** Each image is capped at 3.75 MB decoded (~5 MB of base64). Note this interacts with
  `max_body_bytes`: the 10 MiB default comfortably fits one image, but a multi-image turn may need
  it raised.
- **Errors name the offender.** A bad attachment returns `422` with a detail like
  `` `images[1]` is invalid: unsupported image format ``.

### History

A session's conversation is told two ways. Its **history** is what was said: every turn, the ones a
compaction summarized included, each once where it was said, without what a rewind took back. Its
**context** is what the model sees now: the latest summary, the turns it kept and what came after,
which compaction, repair, image redaction and rewind rewrite. Every row of the session's event log
has a **position**, which is never reused and orders the history; the conversation's position is
its last row's.

```
GET /v1/sessions/{id}/history                     The last page
GET /v1/sessions/{id}/history?before=<position>   The page below a position, toward the start
GET /v1/sessions/{id}/history?after=<position>    The page above a position, toward the end
```

The answer is `{"session_id", "position", "items", "next_before", "next_after"}`, under the
conversation's [`ETag`](#editing-the-conversation). `position` is the conversation's as the page
was read, and `items` are oldest first within the page. A page read below a position, or with no
edge, carries `next_before` exactly while older items remain, the `before` of the page below; a page
read above a position carries `next_after` exactly while newer ones remain, the `after` of the page
above. `limit` is how many items at most, 200 by default and clamped to 1..1000, and `before`
beside `after` is a `422`. A page reads past the rows it leaves out, so it is short only at the end
of the log. The route needs `sessions:r`, reads any session, a sub-agent's included, answers `404`
for an unknown one, and never revives an evicted session.

Each item carries `position`, `created_at` (RFC 3339), `turn_id`, the turn that said it as
`turn.started` announced it and `GET /v1/sessions/{id}/turns` lists it (omitted on a row no turn
said), and a `type`:

| `type` | Fields | What it is |
|--------|--------|------------|
| `message` | `role`, `content` | A message, where it was said |
| `compaction` | `summary`, `checkpoint`, `trigger`, `source`, `memories_written`, `kept_recent`, `generation` | A compaction, where it happened. `summary` is the summary as written, a string; `checkpoint` is the rounds its checkpoint turn ran, as messages with `role` and `content`, empty when none ran; the rest are as [`context.compacted`](#event-types) carries them |
| `rewind` | `ranges`, `turns` | A rewind: the positions it took out of the history, as `ranges` of `{"from", "through"}`, and how many turns they held |
| `join` | `ranges` | A prompt a turn took back to join a message to it, at the positions its `ranges` hold; the message it joined follows |

The history leaves out the rows a rewind took, and the `rewind` item says which, so a client holding
older pages drops its items whose position falls in one of its `ranges`, both ends included;
positions are the store's, so another session's may fall between and are no part of a range. A turn
a compaction kept is taken where it was said, before that compaction, so a rewind reaching back into
the kept turns carries a range for them beside the one for what came after. The history leaves out
the copies of the recent turns a compaction keeps after its summary, which the turns themselves
already show, and the rows that only rewrite the context (a repair, a redaction). A prompt withdrawn because its turn produced
nothing stays in the history, as the screen shows it.

A user message carries what the user typed as a `text` block. Ahead of it, when meka added one,
sits a `turn_context` block: the permission and environment context, catalog changes and resume
notice meka injected for that turn, which the model saw as text ahead of the words. Everything
else meka writes into a user message keeps its own kind, so a client can show it for what it is
and never has to parse a header: a `nudge` (meka's "keep going", with its `kind`); an `inbox`
block for a message that reached the model through the inbox, with `body`, the person's words,
and the `item_id`, `class` and `source` where they were recorded; a `scheduled` block for a
job's fire, with `job_id` and `prompt`, the job's words; and a `background_report` for finished
background work, with `task_ids`. What a person sent where meka expanded part of it keeps its
kind too: an `invocation` for a skill or an MCP prompt invoked by name, with its `kind` (`skill`
or `mcp_prompt`), the `name` invoked (an MCP prompt as `server:prompt`) and the `directive`
typed with it, blank when none was; and a `resource` for a file an editor attached, with its `uri`
and `name`, in the place among the `text` blocks where it was attached. A file attached to a
prompt that invokes a skill is part of the invocation instead: its `directive` names the file by
its mention (`@name`) where it was attached, its `text` holds the contents, and the message has no
`resource` block for it. On each of these `text` is what the model read, header, expanded body or
file contents included, as the `nudge` and `turn_context` blocks hold theirs. An assistant message
carries `text`, `thinking`, `redacted_thinking` (a marker, without the encrypted payload) and
`tool_use` blocks, and the user message after it the `tool_result` blocks that answered the calls.

An image block, whether an attachment on a user message or a tool result, carries its `media_type`
and the `hash` of its bytes rather than the bytes. `GET /v1/sessions/{id}/blobs/{hash}` returns them
under that media type, and only for a session whose messages reference the hash, and a later turn
names the image again by that hash; see [Image attachments](#image-attachments). An image the
request budget redacted to fit the profile's `max_request_bytes` reads in the context as a `text`
block saying so, since that is what the model was last sent; the history shows the image as it was
sent. A tool result may also carry a `tool_reference` item, `{"type": "tool_reference",
"tool_name": "..."}`: a deferred tool that `tool_load` or `tool_search` made callable on a profile
whose [`deferred_tools`](../configuration/config-file.md#deferred_tools) is `native`, which the
provider expanded into the tool's schema where the item sits.

**A transcript is the history plus the feed.** A client that shows a session reads the history's
last page, pages back with `before` as the reader scrolls, and follows the [session
feed](#the-session-feed) for what comes after. When a turn's terminal arrives, what the turn wrote
is in the history up to the terminal's `position`, and `GET /history?after=<the last position the
client holds>` hands it over as stored.

### Context

```
GET /v1/sessions/{id}/context            What the context holds, measured
GET /v1/sessions/{id}/context/messages   Every message of the context
```

`GET /v1/sessions/{id}/context/messages` answers `{"session_id", "position", "messages"}`: every
message of the context, in the order the model reads them, unpaged, since the context is one
request's worth. It answers under the conversation's `ETag`. Each message has `role` and `content`,
typed as the history types them, and the `position`, `created_at` and `turn_id` of the history item
it shows: its own row; for a copy a compaction kept, the row it was said in, however many
compactions in a row kept it, so a kept turn shows the turn and time it was said, and once turn
retention has taken that row, the copy's own position, the history item it then is, under the same
turn and time; for a message meka rewrote in place (an image it left out, a retry's degraded
copy), the row it was said in, under that row's turn and time; and for a summary, its compaction's
position, with no `turn_id`. A summary also carries `compaction`, the
`trigger`, `source`, `memories_written`, `kept_recent` and `generation` its history item has; no
other message carries it.

The context is what the model sees, not a transcript. The turns a compaction kept are in both, so a
transcript drawn from the history and the context together shows them twice: draw one from the
history and the feed, and read the context to show what the model is working from.

`GET /v1/sessions/{id}/context` answers `{"session_id", "position", "used", "window", "overhead",
"compact_at_percent", "generation"}`: the tokens behind the most recent provider round, the model's
context window, the estimated system prompt and tool schemas (the part of the window compaction
cannot reclaim), the occupancy auto-compaction fires at, and how many compactions the session has
had. `used` and `overhead` are omitted until a turn has run since this process loaded the session,
`window` when meka has no metadata for the model, and `compact_at_percent` when auto-compaction is
off. What the session has spent across its turns is `usage` on the [session record](#sessions).

None of the history, the context and `GET /v1/sessions/{id}/tools` loads an evicted session.
Reading is not permitted to take the session's cross-process lock, which a write would hold for
`idle_timeout`. The history and the context answer from the store, `/context` with the live
counters omitted; `/tools` returns 409, since a catalog needs a loaded session.

### Editing the conversation

Every change to the conversation writes a row to its event log, so its position moves on each: a
message appended, a compaction, a rewind, and the rewrites a turn makes of its context, a repair of a
malformed request, the redaction of an image that no longer fit the request size budget, a prompt
taken back to join a message to it or withdrawn because its turn produced nothing (for a recurring
job, or a turn sent with `options.unanswered_message` set to `withdraw`). Three things make editing
the conversation safe:

- **`ETag` and `If-Match`.** `GET /history` and `GET /context/messages` answer with the
  conversation's `ETag`, its position quoted (`"4127"`), and so do the answers of `POST /rewind`
  and `POST /compact`, with the position they leave. Send the tag from whichever listing was read
  as `If-Match` on `POST /rewind` or `POST /fork`, and an edit decided on a conversation that has
  since changed is refused with `412` `/errors/precondition-failed`, whose `position` is the
  current one; read again and decide again. Without the header the call is unconditional.
- **`position` on every terminal.** `turn.finished`, `turn.failed` and `turn.canceled` carry the
  conversation's position as the turn ends. What the turn wrote is in the history up to there, so a
  client that rendered the turn live reads `GET /history?after=<the last position it held>` to
  hold it as stored.
- **`conversation.rewound`** on the [session feed](#the-session-feed) carries the `rewind` item
  as the history lists it, so every reader learns what left the history and the position its next
  edit needs. It is emitted outside any turn and carries no `turn_id`.

A client that shows the history needs nothing more: a compaction is an item appended to it, and a
rewind an item saying which items to drop. A client that shows the context reads
`GET /context/messages` again after `context.compacted`, `conversation.rewound` and each terminal,
since a repair, a redaction or a withdrawal rewrites the context inside a turn with no event of its
own.

### Compaction, rewind and export

`POST /v1/sessions/{id}/compact` summarizes the conversation now. The body is optional:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `instructions` | string | *(none)* | Guidance on what to keep or drop, as `/compact <instructions>` on the screen |
| `keep_recent` | bool | *(meka decides)* | Whether to keep the most recent turns verbatim after the summary |

The answer is the `compaction` item as the history lists it, under the conversation's new `ETag`.
The compaction is a turn of its own: the feed announces it with a `turn.started` whose `source` is
`"compaction"`, under a `turn_id` equal to the compaction's `compaction_id`, and carries
`context.compacting` and `context.compacted` inside it; `GET /v1/sessions/{id}/turns` lists it with
what it spent, and the item names it in `turn_id`. A compaction that runs inside a turn, automatic
or asked for by the agent, is that turn's, and so is what it spent.

`POST /v1/sessions/{id}/rewind` takes the conversation back to how it stood before a turn. The body
is required:

| Field | Type | Description |
|-------|------|-------------|
| `turn_id` | string | The first turn to drop, as `GET /v1/sessions/{id}/turns` or a history item's `turn_id` names it |

That turn and every later one leave the context, and the history gains a `rewind` item for them. A
turn the context does not hold, another session's, one a compaction summarized, a compaction's own
or one that left nothing in it, is a `422` naming it. A turn a compaction summarized in part, its
prompt in the summary and its later rounds kept, is rewound from its first kept round: the summary
goes on holding what it summarized. An `If-Match` header holding the conversation's `ETag` makes the
rewind conditional on the conversation being as it was read, and a `412` says it is not; see
[Editing the conversation](#editing-the-conversation). The answer is the `rewind` item as the
history lists it, under the conversation's new `ETag`, and the feed carries `conversation.rewound`.
A retried rewind names a turn that is gone, and is refused with that `422` rather than taking
another. The screen's `/rewind` and `meka session rewind` count turns back from the end, and stop at
a compaction's summary as this route does.

`GET /v1/sessions/{id}/export` returns the session's archive as JSON, the object
`POST /v1/sessions/import` and `meka session import` take, versioned by its `format_version`. A
Markdown transcript is `meka session export`'s; over HTTP, render the history.

## Blocking response

With `stream: false` (the default), the server holds the connection until the turn completes, then
returns a single JSON response: the turn record, as `GET /v1/sessions/{id}/turns/{turn_id}` returns
it, with what the turn wrote and what meka said about it.

```json
{
  "id": "7f3b...",
  "source": "client",
  "started_at": "2026-05-26T13:46:40Z",
  "permission": "workspace",
  "ended_at": "2026-05-26T13:47:01Z",
  "status": "succeeded",
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 12340,
    "output_tokens": 567,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 8000
  },
  "messages": [
    {
      "position": 4121, "created_at": "2026-05-26T13:46:40Z", "turn_id": "7f3b...",
      "type": "message", "role": "user",
      "content": [
        {"type": "turn_context", "text": "[Context]\n..."},
        {"type": "text", "text": "What is in src?"}
      ]
    },
    {
      "position": 4122, "created_at": "2026-05-26T13:46:44Z", "turn_id": "7f3b...",
      "type": "message", "role": "assistant",
      "content": [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "tu_1", "name": "file_read", "input": {"path": "src/main.rs"}}
      ]
    },
    {
      "position": 4123, "created_at": "2026-05-26T13:46:45Z", "turn_id": "7f3b...",
      "type": "message", "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": "tu_1", "is_error": false,
         "content": [{"type": "text", "text": "fn main() { ... }"}]}
      ]
    },
    {
      "position": 4124, "created_at": "2026-05-26T13:47:01Z", "turn_id": "7f3b...",
      "type": "message", "role": "assistant",
      "content": [{"type": "text", "text": "Here are the files in src/: ..."}]
    }
  ],
  "notices": []
}
```

Key fields:

- **The record**: `id`, `source`, `started_at`, `permission`, `ended_at`, `status`, `stop_reason`,
  `error`, `reason`, `profile` and `usage`, as [`last_turn`](#sessions) describes them.
- **`messages`**: the history items the turn wrote, in order and as
  [`GET /v1/sessions/{id}/history`](#history) lists them: first the message the turn opened with,
  your words behind the `turn_context` block meka wrote and any report that rode them, then the
  assistant's messages with their text, their thinking whenever it was recorded, and their tool
  calls, the tool-result messages that answered those calls, and a compaction that ran mid-turn as
  its `compaction` item. The reply a bot shows is the text blocks of the assistant messages,
  joined. `supports_reasoning_stream` governs only the `thinking.delta` events.
- **`notices`**: provider advisories and warnings about approvals refused without asking.
- **`refusal_text`**: present only when `stop_reason` is `"refusal"`.

## Streaming response

Every resident session has one event feed. Everything a turn emits goes on it, whoever started the turn: a `POST /turns`, a scheduled fire, a background outcome, an inbox item. `POST /turns` with `stream: true` answers with a view of that feed scoped to the one turn it started, as `text/event-stream`, and closes after the turn's terminal. `GET /v1/sessions/{id}/feed` is the feed itself, across turns; see [The session feed](#the-session-feed). Every event has a monotonic `id`, a named `event` type, and a JSON `data` payload, and every payload carries `turn_id` and `session_id`, so a client holding several feeds can file an event without per-connection state.

### Event types

#### Lifecycle

| Event | Payload | When |
|-------|---------|------|
| `turn.started` | `turn_id`, `session_id`, `started_at`, `source` (`"client"`, `"inbox"` with `item_ids`, `"schedule"` with `job_id`, `"background"`, `"compaction"` for the checkpoint turn `POST /compact` runs, or `"parent"` with `parent_id` and `tool_call_id` on a sub-agent's feed) | Turn begins |
| `message.appended` | `role`, `content`, `turn_id`, `session_id` | A user-role message carrying words entered the conversation: the message the turn opened with, whoever opened it, or the one an interrupt appends after a cut answer. `content` is the message's blocks as `GET /history` types them (`turn_context`, `text`, `invocation`, `resource`, `scheduled`, `background_report`, `inbox`, `image` by `hash`) |
| `turn.finished` | `turn_id`, `session_id`, `stop_reason`, `usage`, `position`, optional `refusal_text`, `profile` when a [fallback profile](sessions.md#fallback-profiles) served the turn | Turn completed successfully |
| `turn.failed` | `turn_id`, `session_id`, `error` (Problem Detail shape), `position`, `message_withdrawn` when the turn began, `profile` when a fallback profile served the turn | Turn failed mid-stream |
| `turn.canceled` | `turn_id`, `session_id`, `reason` (`"client"` for `POST /cancel`, `"server_shutdown"`, `"abandoned"` when a held turn's request had been gone for the grace with nobody else listening, `"sse_lag"` when the only consumer fell behind and the turn was stopped for it, or `"parent"` on a sub-agent's feed, whose turn stops with its parent's), `position`, `message_withdrawn` when the turn began, `profile` when a fallback profile served the turn | Turn was canceled |
| `turn.nudged` | `kind`, `text`, `turn_id`, `session_id` | The turn went on past the model's reply: meka wrote `text` into the conversation as a user-role message to send the model back to work, `kind` `visible_reply` after a reply with no visible text, `checklist` after one that left checklist items open |
| `task.reported` | `task_ids`, `text`, `turn_id`, `session_id` | A turn in flight read finished background work at a tool-round boundary: meka wrote `text`, the report the model reads, into the round's user-role message behind its results |

`turn.finished`, `turn.failed`, and `turn.canceled` are **terminal** for the turn: a `POST /turns` stream closes immediately after its own, and the feed carries on to the next turn. `turn.failed` and `turn.canceled` also carry `message_withdrawn` when the turn began, whether it took the message it was sent back out of the conversation; see [Resending a failed turn](#resending-a-failed-turn). All three carry `position`, the conversation's position as the turn ends: what the turn wrote is in the history up to there, so a client that rendered it live reads `GET /history?after=<the last position it held>` to hold it as stored. It is omitted only when the store could not say, and on [a sub-agent's feed](#a-sub-agents-feed); see [Editing the conversation](#editing-the-conversation).

`message.appended` is how a reader that did not open a turn learns what it answers: a scheduled job's fire, a background report, another client's words, an inbox item. It follows the turn's `turn.started` and precedes every delta, tool event and `inbox.delivered` of that turn; the message an interrupt appends follows the `notice` announcing the cut. The feed mirrors the conversation at two grains: a message that enters whole is announced whole by this event, and a block added to a message a round is still building is announced by the event that names it. `task.reported` and `turn.nudged` carry the text meka wrote; `inbox.delivered`, which also covers an interrupt joined onto an opening nothing had answered, carries the items' ids alone, since the words are the producer's, and a reader that is not the producer takes them from `GET /history`. An `inbox` block in `message.appended` says the item was appended, as its own `appended` state says; `inbox.delivered` is the moment the model read it. `message_withdrawn: true` on a turn's terminal is a fact about the context: the opening stays in the history, as the screen shows it, and a client showing the context drops it. A recurring job's fire, or an inbox turn retried after a failure, appends its opening on every attempt and takes it back out of the context each time it ends with nothing from the model. The event carries the `turn_context` block as the history does, so a reader's replay ring grows with it.

#### Inbox

| Event | Payload | When |
|-------|---------|------|
| `inbox.delivered` | `item_ids`, `turn_id`, `session_id` | The provider accepted a request carrying these items, so the model has read them |
| `inbox.deferred` | `item_ids`, `session_id`, `not_before`, `error` (Problem Detail shape) | A turn on these items failed in a way waiting can change; the next attempt runs at `not_before` |
| `inbox.failed` | `item_id`, `session_id`, `reason`, `error` (Problem Detail shape) when a failed turn decided it | meka gave up on the item and withdrew it: its retry ceiling passed, or its turn failed in a way no retry would change |
| `inbox.withdrawn` | `item_id`, `session_id` | A client withdrew the item with `DELETE`, or the turn it opened was canceled; `turn_id` names the turn in flight at the time, if any |

`inbox.delivered` arrives inside the turn that read the item, before that turn's terminal: for a `steer` read at a round boundary, after the `tool_call.completed` of that round; for an `interrupt` that cut the answer, after the `notice` announcing the cut. See [The inbox](#the-inbox).

#### Content deltas

| Event | Payload | When |
|-------|---------|------|
| `assistant_text.delta` | `text` | Each chunk of assistant text |
| `thinking.delta` | `text` | A chunk of extended thinking content (only when `supports_reasoning_stream: true`) |

Reasoning streams in chunks, one event per chunk, the way `assistant_text.delta` does; concatenate them to reassemble the block. A turn the provider answered without streaming sends the block as a single delta, so a client never has to tell the two apart. The blocking response (`stream: false`) reports each block whole, as a `thinking` content block in `messages`, whenever the turn recorded one; `supports_reasoning_stream` governs only these deltas.

#### Tool execution

| Event | Payload | When |
|-------|---------|------|
| `tool_call.composing` | `id`, `name` | The model started writing the call's arguments |
| `tool_call.executing` | `id`, `name`, `input`, `display_summary` | Tool call starts |
| `tool_call.completed` | `id`, `is_error`, `content`, `change` | Tool call finishes |
| `tool_call.progress` | `id`, `progress`, `total`, `message` | An MCP tool reported progress while running |
| `tool_call.output_delta` | `id`, `chunk` | A running `shell_execute` produced output; append `chunk` to what you show for the call |

`tool_call.progress` relays an MCP server's `notifications/progress` for a call that is still running: `id` is the call's, as its `tool_call.executing` named it, `progress` the server's counter, `total` its target when it gave one, and `message` its text. `total` and `message` are omitted when the server did not send them. A call in the background reports no progress, its call having returned. Only MCP tools report progress; a built-in's next sign of life is its `tool_call.completed`, except `shell_execute`, whose output streams as `tool_call.output_delta`.

`tool_call.output_delta` is progress rather than history, and the feed treats it so: it carries no `id`, is never replayed after a reconnect, and never displaces the events a `Last-Event-ID` resumption depends on. Command output is coalesced to about one event per 150 ms per call, whatever is left is flushed just ahead of the call's `tool_call.completed`, and that event still carries the whole output. A command run with `background: true` is not streamed: its call returns at once with a task id, and its output arrives with the task's outcome. What a sub-agent under an `agent_spawn` call is doing is not summarized on the parent's feed: it is [the sub-agent's own feed](#a-sub-agents-feed), every call in full.

`change` is present on the completion of a call that changed a file, a memory or a skill (`file_write`, `file_edit`, `memory_write`, `memory_edit`, `memory_delete`, `skill_write`, `skill_edit`, `skill_delete`) and absent otherwise, a failed call included. It carries the whole text before and after as `old` and `new`, with `path` for a file or a skill's `SKILL.md`; a memory's text is what `meka memory export` writes. `old` is omitted when the call created the entry and `new` when it deleted it, so a client can draw the diff itself or show what the entry now says. It is live only: the log keeps no result metadata, so the history never carries it, and a reconnect that has fallen past the replay ring cannot recover it.

The arguments are written between `tool_call.composing` and `tool_call.executing` on the same `id`, which makes that interval the only thing on the stream that separates the agent *writing a message* from the agent doing anything else. Assistant text is usually narration around a call rather than the reply itself, and by `tool_call.executing` the arguments are already finished. A client drawing a typing indicator for a tool like an MCP `send_message` raises it on the first and drops it on the second. The payload is the id and the name because nothing else has streamed yet: which conversation a message is for is not known until `tool_call.executing`.

Three limits. The event exists only when meka streams from its provider, so a server started with `--no-stream` receives each call whole and emits `tool_call.executing` with nothing before it. The pairing is not guaranteed, because a turn that fails or is canceled mid-call emits `tool_call.composing` with nothing after it, and a round cut, or ended on an error the turn recovers from, leaves a `tool_call.composing` or a `tool_call.executing` with no completion while the turn goes on, sometimes followed by a later call under the same `id`; so close per-`id` state on the terminal event as well, and take a later `tool_call.composing` for an `id` as a new call. And the interval is only wide on backends that stream a call as it is written (`anthropic-messages`, `claude-subscription`, `openai-responses`, `chatgpt-subscription`); `openai-chat-completions` resolves each call's name and arguments together when the stream ends, so there the two events arrive back to back.

#### Notices and pauses

| Event | Payload | When |
|-------|---------|------|
| `notice` | `level`, `text` | Provider advisories or warnings |
| `approval.requested` | `request_id`, `tool_name`, `input`, `expires_in_seconds`, `expires_at`, `subagent_id` when a sub-agent asked | Permission approval needed (approvals on, call above the level) |
| `approval.resolved` | `request_id`, `outcome` (`allow`, `deny`, `allow_always`, `deny_always`, `expired` or `canceled`) | The parked prompt closed, however it closed |
| `elicitation.requested` | `request_id`, `server`, `kind` (`form` or `url`), `message`, `schema` on a form, `url` on a URL, `expires_in_seconds`, `expires_at`, `subagent_id` when a sub-agent's server asked | An MCP server asked the person something, on a session with `supports_elicitation` |
| `elicitation.resolved` | `request_id`, `outcome` (`accept`, `decline`, `cancel`, `expired` or `canceled`) | The parked elicitation closed, however it closed |

#### Context

| Event | Payload | When |
|-------|---------|------|
| `context.compacting` | `compaction_id`, `trigger` | A compaction began; the checkpoint turn or the summarizer is about to run |
| `context.compacted` | `compaction_id`, `trigger`, `source`, `memories_written`, `kept_recent`, `generation` | The conversation was summarized and the context replaced |
| `context.compaction_failed` | `compaction_id`, `trigger`, `error`, `interrupted` | The compaction that began under `compaction_id` ended without replacing the window |
| `context.measured` | `used`, `window` when known | What is in context now, the figure `GET /v1/sessions/{id}/context` answers with: after every provider round, and after a compaction or rewind with the estimate it leaves |
| `conversation.rewound` | `session_id`, and the `rewind` item as the history lists it: `position`, `created_at`, `type`, `ranges`, `turns` | `POST /rewind` took turns out of the context; emitted outside any turn, so it carries no `turn_id` |
| `checklist.updated` | `items`, the whole open list as the session record carries it | A `checklist_*` call changed the list; empty once everything is disposed of |

A turn that stops with a checklist item open does not end: meka sends the model back to the list
and the turn carries on, inside the same `turn_id`, with a `turn.nudged` carrying the message it
wrote, a `notice` for each nudge (`Checklist: 2 open items, continuing, nudge 1 of 3`) and a `warn`
notice when a cap ends the turn with items still open: three nudges in a row without a tool call,
or nine in the turn. The message is a user-role message of the conversation whose first block is a
`nudge` block, so the history shows it between the reply it answered and the reply after it,
under the same `turn_id`, and a client can tell it from the person's words; a blocking
`POST /turns` waits through those rounds and carries it among the turn's `messages`. A steer that
arrives while the model writes the reply that gets nudged rides the nudge as a text block behind
it, the way one rides a tool round's results, and `inbox.delivered` follows. See
[Checklist](checklist.md).

`context.compacted` is the one event on this stream that rewrites the context rather than adding to it: the messages the summary replaced leave what the model sees. The history loses nothing and gains the compaction's item at its position. `trigger` is what set it going: `reactive` (a measured occupancy crossed the ceiling), `proactive` (the turn's projected request would have), `manual` (`/compact` or `POST /compact`), `requested` (the agent's own `context_compact`), `emergency` (the provider refused the request as too large) or `retention` (a turn aged past `turn_retention`). `source` is `checkpoint` or `summarizer` (they differ in fidelity, not just mechanism), `memories_written` names what the checkpoint turn wrote, `kept_recent` says whether the recent turns were kept verbatim after the summary, and `generation` counts compactions from 1. `context.compacting` opens the compaction under the same `compaction_id` before any request is made, since a checkpoint turn can take a while; `context.compaction_failed` closes one that ended without replacing the window, `interrupted` when the user stopped it.

The same facts, without `compaction_id`, are on the compaction's [history](#history) item, beside its summary and checkpoint, and on its summary in [`GET /context/messages`](#context) as `compaction`.

### Heartbeats

A `: keep-alive` comment is sent every 20 seconds. SSE clients ignore these automatically. The stream also sends `retry: 3000` as its first line, hinting clients to reconnect after 3 seconds on disconnect.

### Falling behind

A stream's events are broadcast to every reader of the session, and the server buffers up to 256 ahead of a slow one. A reader that falls behind is caught up from the replay ring, the same replay a reconnect with `Last-Event-ID` gets, done by the server on the reader's behalf: it sees a pause and nothing else. When the ring no longer reaches back to what the reader last received, the hole is said as a `feed.gap` event, below, and the stream carries on.

One case ends the stream instead. A `POST /turns` with `stream: true` is run for its client, so when that client is the only reader and falls behind, the turn is canceled to stop burning provider tokens, and the stream ends with a terminal `turn.failed` carrying error type `https://meka.bot/errors/sse-lag`. That event is the stream's, sent before the turn has unwound, so it carries no `message_withdrawn`; the outcome recorded for a later re-attach is a `turn.canceled` with `reason: "sse_lag"` and does. Retry by submitting a new turn. A turn is never canceled for a reader of the [session feed](#the-session-feed) or the [server feed](#the-server-feed), because the turn was not run for it.

#### The feed itself

| Event | Payload | When |
|-------|---------|------|
| `feed.gap` | `session_id` on a session's feed, `dropped` when the feed can count the hole | This reader's copy of the feed has a hole: the replay ring no longer reaches the position it resumed from, or what it fell behind by. Read `GET /v1/sessions/{id}/history?after=<the last position held>` (or `GET /v1/sessions` for the server feed) to fill it. Carries no `id`, since it is the reader's and not the session's |

`dropped` is how many numbered events the hole holds. It is absent when the reader resumed from an id this feed never issued, after a restart say, since nothing says how far back it stood.

### The session feed

`GET /v1/sessions/{id}/feed` is the session's feed: every event of every turn, across turns, for as long as the connection is held. It is how a client sees the turns nobody asked for over HTTP, a scheduled job firing at three in the morning or a background task reporting back, and it is where a client that submits through the [inbox](#the-inbox) learns what became of its items. Subscribe once, and file events by the `turn_id` they carry. Loading the session is part of opening the feed, so a bridge can subscribe before it has anything to submit, and a reconnect to an evicted session gets its feed back rather than a 404. That loading takes `sessions:w`, the scope that drives a session: a revived session is held in memory and under its cross-process lock for as long as the stream stays open, so a token with `sessions:r` alone attaches to a session this process has loaded and is answered `409` (`/errors/session-not-loaded`) otherwise. Such a reader keeps nothing loaded either: the idle timeout evicts the session past it, which closes its stream, and it reconnects to that same `409` until a writer loads the session again.

Send the last id you received as a `Last-Event-ID` header (browser `EventSource` does this automatically) or as a `?last_event_id=` query parameter, and the server replays what you missed before following the live feed.

Add `?attend=true` to say that this reader shows approval prompts and answers them. It needs `sessions:w`, and while at least one attending reader is connected a gated call on any turn parks as `approval.requested` instead of being refused without asking; when the last one disconnects, a parked prompt is canceled. See [Approvals](#approvals). A reader that attends also counts as a renderer of reasoning deltas, so on a session with `supports_reasoning_stream` its turns lose their retry the way a streaming client's do.

```bash
curl -N -H "Authorization: Bearer $TOKEN" \
     -H "Last-Event-ID: 1793001600000042" \
     "http://localhost:8080/v1/sessions/$SESSION/feed"
```

Ids run across the whole session and the ring spans turns, so an id from an earlier turn is an ordinary position: everything after it replays, the later turns' terminals included. When a turn is in flight as you attach, the feed opens with a `turn.started` carrying `"resumed": true` and the `turn_id` you joined, which is how to tell "my stream resumed" from "a newer turn started while I was away"; that event is synthesized rather than replayed, so it carries no `id:` and no `started_at`, but it names the turn's `source` (with its `item_ids` or `job_id`). The turn's `message.appended` follows it the same way, marked `"resumed": true` and without an `id:`, for a reader that has not had it: one joining, or one resuming from behind it after the ring let it go. A reader the replay hands it to, or one resuming from past it, is not handed it again. So a client joining mid-turn reads the reply under the message it answers; it is the opening as first announced, and what the turn appended since is read from `GET /history`. Everything after it is the real thing. With no turn in flight there is nothing to re-issue, and the most recent turn's terminal is handed over when the ring no longer holds it, so a client that reconnects late still learns the outcome.

**The feed does not end with a turn.** A client that wants one turn's outcome stops reading at that turn's terminal, or holds `POST /turns` with `stream: true`, whose stream closes after its own turn.

Three limits, all deliberate:

- **The replay buffer is bounded** by `[serve] stream_replay_events` (default 256). If your `Last-Event-ID` is older than the oldest retained event, the stream opens with a `feed.gap` rather than a transcript that silently skips; see [Falling behind](#falling-behind). Read `GET /history?after=<the last position held>` to fill it.
- **Only the most recent turn's terminal is retained** past the ring. Everything else a late client needs is in `GET /history`.
- **A held turn is not canceled immediately when its client disconnects.** It keeps running for `[serve] stream_reattach_grace` (default 30s) waiting for a listener to come back; after that the agent loop stops at its next round boundary, since nobody is listening, and the turn ends with `turn.canceled` and `reason: "abandoned"`. Set `"0s"` to stop it at the first boundary after the drop. The rule is only for turns a request holds, `POST /turns` without `detach` in either encoding: a detached turn, or one the server started for a fire, an outcome or an inbox item, runs for the session and is never stopped for want of a reader. A feed subscriber is a listener, so a held turn with one attached runs on.

A session with a live feed subscriber is not idle, so the [idle sweep](#idle-timeout-and-gc) leaves it resident. Opening the feed loads the session if it was not, exactly as submitting a turn does, so a `sessions:r` token can bring one into memory and keep it there, and the route answers `409` `session-locked` where `POST /turns` would. A sub-agent's id is the one id the route never loads: it is the sub-agent's own feed while its parent runs it, below.

### A sub-agent's feed

`GET /v1/sessions/{id}/feed` on the id of a sub-agent this process is running is that
sub-agent's feed: the same events a session feed carries, every one of its tool calls in full,
with its own ids, ring and `Last-Event-ID`. The feed exists for exactly as long as the parent's
spawn or follow-up runs the sub-agent, the span its record reports `turn_in_flight`; a client
that sees a sub-agent in flight, on a listing or from a `session.updated` on
[the server feed](#the-server-feed), opens the feed with the same call it uses for any session.
Its openings are announced as any session's: the `message.appended` after its first `turn.started` carries the words the parent's spawn or follow-up gave it.
It opens with a `turn.started` whose `source` is `"parent"`, with the parent's `parent_id` and
the `tool_call_id` of the `agent_spawn` or `agent_followup` call running it, and closes with the
terminal, which carries no `position`. A sub-agent's turn ends when its parent's does: a parent
canceled by a client, or stopped with the server, ends the sub-agent's with `turn.canceled`
and `reason: "parent"`.

The feed is read, never attended: a prompt a sub-agent's call parks is on its parent's feed,
where the one answerer is, and is answered on the parent's `responses` route. The sub-agent's
feed mirrors the `approval.requested` and the `approval.resolved`, with `subagent_id`, so a
client watching the sub-agent sees it waiting and knows where to answer. `?attend=true` on a
sub-agent's feed is refused with `422` `session-not-drivable`.

Once the run ends the feed is gone, and the route answers `409` `subagent-not-running`, naming
the parent whose run it belongs to. What the sub-agent did is in its history, `GET /history` and
`GET /turns` on its own id, and `last_turn` on its record says how the run ended. A sub-agent a
different process runs, a screen on the same store say, has no feed here: the same limit
`turn_in_flight` has.

### The server feed

`GET /v1/feed` is the listing's change feed: one connection, across every session this
process holds, carrying exactly the events that change a session record and nothing a session
feed is for. A client keeping a list of sessions keeps it current from this and opens a session's
own feed only for the one it is looking at. It needs `sessions:r`, numbers its events on its own,
and replays from `Last-Event-ID` (or `?last_event_id=`) out of a ring of its own, with the same
`feed.gap` when the ring no longer reaches back far enough and the same catch-up for a reader that
falls behind.

| Event | Payload | When |
|-------|---------|------|
| `session.created` | the session record, as `GET /v1/sessions/{id}` answers | A session was created, forked, imported, or spawned as a sub-agent by this process |
| `session.updated` | the session record | A `PATCH` changed it, or a sub-agent's run on it began or ended |
| `session.deleted` | `id`, `session_id` | A `DELETE`, or the idle sweep with `delete_on_idle` |
| `turn.started`, `turn.finished`, `turn.failed`, `turn.canceled` | as on the session feed | Every turn on every session this process holds, deltas left out; a sub-agent's run reaches this feed as the `session.updated` on its record |
| `approval.requested`, `approval.resolved`, `elicitation.requested`, `elicitation.resolved` | as on the session feed | A prompt was parked or closed, which moves `requests_pending` |

A `session.updated` carries the whole record so a client replaces its row rather than patching
it. The same `session.updated` goes to the session's own feed, so a client on one session also
learns its title or profile moved under it. The feed sees this process's changes only: a session
another process renamed or deleted, a screen open on the same store say, is not announced, the
same limit `turn_in_flight` has.

## Webhooks

`meka serve` can POST to configured endpoints when something happens that no client is necessarily waiting on: a scheduled job firing overnight, a background task finishing long after the turn that started it.

```toml
[[serve.webhooks]]
url = "https://bridge.example/meka-hook"
secret = "${MEKA_WEBHOOK_SECRET}"     # or secret_file = "/etc/meka/hook.secret"
events = ["turn.finished", "turn.failed", "task.finished", "schedule.fired", "inbox.deferred", "inbox.delivered", "inbox.failed"]
timeout = "10s"                        # per attempt, default 10s; "0s" is refused at startup
max_retries = 3                        # after the first attempt, default 3, at most 10
```

`events` is required and every name must be recognized: an endpoint whose only subscription is a typo would be silently never called, so an unknown event is a startup error rather than a warning.

`turn.finished` and `turn.failed` cover turns submitted through `POST /turns`. A scheduled job's turn fires `schedule.fired` (which carries its own `status`) instead, so no turn produces two deliveries. A turn the server runs purely to report a background outcome fires neither: the news is the task's, and `task.finished` has already carried it.

`task.finished` is not a turn event. It fires when a background task reaches a terminal state, whether or not any turn reports it: a canceled task fires it with no turn at all, and its outcome then rides the next round boundary of a running turn or whichever turn the session takes next. Expect it alongside a `turn.finished` when a client's own `POST /turns` is what carries the outcome, and expect it on its own for a task interrupted by a host that died, which no turn ever ran.

`inbox.deferred`, `inbox.delivered` and `inbox.failed` are the [inbox](#the-inbox)'s three outcomes: an item waits for another attempt, the model read it, or meka gave up on it. The `error` on a webhook copy of `inbox.deferred` or `inbox.failed` is the Problem Detail's `type`, `title`, `status` and `detail`; `provider_response` rides the feed's copy alone. A turn the server runs on inbox items posts `turn.finished` or `turn.failed` like a client's, since nothing else carries it.

A client that wants to know about everything the agent did should subscribe to all seven.

### Payloads

Every delivery carries `delivery_id`, `event`, `timestamp`, and event-specific identifiers:

```json
{
  "delivery_id": "6c1f...",
  "event": "schedule.fired",
  "timestamp": "2026-02-01T03:00:00Z",
  "job_id": "9f2c...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed"
}
```

A `schedule.fired` `status` is `completed`, `failed`, `canceled` when the turn was stopped before it finished, as happens when the server drains during it, or `not_run` when the job's session could not be opened for it (its profile left `config.toml`, or the session is gone); a `not_run` fire is tried again when its lease expires, so the status may repeat until the attempts run out.

**Payloads never carry message content.** A webhook URL is a string in a config file: it can be mistyped, it can outlive whatever owned it, and anything that learns it can reach it. So a delivery tells you *what happened to which session*, and you fetch the conversation with your own bearer token over the API you already authenticate against. A compromised endpoint learns that a session was active, not what was said in it.

### Verifying a delivery

When `secret` is set, each request carries `X-Meka-Signature: sha256=<hex>`, an HMAC-SHA256 over `<timestamp>.<body>` keyed with the secret. The timestamp is *inside* the signed material, so a captured delivery cannot be replayed forever: reject anything whose `X-Meka-Timestamp` is too old and the window closes.

Each **attempt** carries its own timestamp and signature. A retry can land minutes after the first attempt, so re-sending the original stamp would have it rejected by that very window. Deduplicate on `X-Meka-Delivery`, which stays constant across a delivery's attempts.

```python
import hmac, hashlib

def verify(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    expected = "sha256=" + hmac.new(
        secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)
```

Deliveries also carry `X-Meka-Event`, `X-Meka-Delivery` (unique per delivery, for deduplicating retries), and `X-Meka-Timestamp`.

`X-Meka-Timestamp` and the body's `timestamp` field differ on a retry, deliberately. The header is when *this attempt* was sent, re-stamped each time, because that is what your replay window is checking; a retry carrying the original time would be rejected as stale by the very check the header exists for. The body's field is when the *event* happened and stays fixed across attempts, so ordering and deduplication see one event rather than several.

Omitting `secret` is allowed for loopback receivers and logs a startup warning; no signature header is sent, rather than one computed over an empty key.

### Delivery semantics

Deliveries are notifications, not a durable queue. They are not persisted, not retried across a restart, and outstanding attempts are abandoned when the process exits: a delivery in flight during a `SIGTERM` is lost. That is the trade for never blocking the work that triggered it. Anything you cannot afford to miss should be reconciled by polling (`GET /v1/jobs`, `GET /v1/tasks?session=<id>`), with the webhook as the fast path rather than the only one.

Delivery is fire-and-forget on a detached task, so a slow or dead receiver never wedges the scheduler behind it. A `5xx` or a transport error is retried with exponential backoff (1s, 2s, 4s, capped at 30s) up to `max_retries`. A `4xx` is not, since retrying cannot fix a request the receiver considers malformed, with two exceptions: `429 Too Many Requests` and `408 Request Timeout` say "not now" rather than "not ever" and are retried like a `5xx`. That matters because several jobs sharing a cron minute deliver as a burst, which is exactly when a receiver rate-limits. `Retry-After` is not honored; the backoff above is used regardless. After the last attempt meka logs one `warn` and gives up. Turn cancellations are not delivered: the client that canceled already knows.

## Permission levels over HTTP

The same four [permission levels](./permissions.md) apply: `none`, `read`, `workspace`, `unrestricted`. Set the level at session creation or update it via `PATCH /v1/sessions/{id}`; `approvals` sits beside it on both, and `{"approvals": true}` in a `PATCH` body turns it on for a live session.

Both apply during a turn: the next tool call is checked against the new level and the new switch, as the screen's Shift+Tab and ACP's `session/set_mode` do, so dropping a running session to `read` is a brake and not a request to stop. `working_directory` and `profile` wait for the turn to end, and a body naming either answers `409` `turn-in-flight` meanwhile.

### Approvals

With `approvals: true`, a tool call above the session's level parks as an `approval.requested` SSE event whenever someone is there to answer it: the client of a `POST /turns` with `stream: true`, held or detached, or a feed reader that opened `GET /v1/sessions/{id}/feed?attend=true` (which needs `sessions:w`, the scope that answers). Any turn qualifies, an inbox turn, a scheduled fire or a background outcome included, so a UI that submits through the inbox and watches the feed is asked like a streaming client is. With nobody attending, the call is refused without asking and a `notice` says so, unless the session declared [`answers_prompts_on_return`](#capabilities), in which case a prompt in a turn no request holds parks and waits for the client to come back. The stream stays open while waiting. Your client resolves it by POSTing to the responses endpoint:

```
POST /v1/sessions/{id}/responses/{request_id}
Content-Type: application/json

{"outcome": "allow"}
```

Possible outcomes:

| Outcome | Effect |
|---------|--------|
| `allow` | Run this tool call |
| `deny` | Refuse this tool call |
| `allow_always` | Allow this and all future calls to this tool (session-scoped) |
| `deny_always` | Deny this and all future calls to this tool (session-scoped) |

`input` is every argument the call was made with, and a prompt should show it: `tool_name` alone asks you to approve a write without showing what is written. If no response arrives within 30 minutes the request is denied; `expires_in_seconds` on the event carries that figure, counted from when the prompt was parked, and `expires_at` carries the deadline itself, which is the one to show when the event reached you from the replay ring minutes after it was parked. It is the same backstop an ACP client's prompt gets. When the last client that could answer disconnects, the prompt waits out [`[serve] stream_reattach_grace`](../configuration/config-file.md#servestream_reattach_grace), the window a held turn already gives its reader, and is canceled only if nobody is attending by then: a browser tab reloading or waking from suspension finds its prompt still parked, and a closed one costs the call after the window rather than at once. On a session with `answers_prompts_on_return`, a prompt in a turn no request holds is not canceled for absence at all; it waits out its expiry. An approved call still runs at the session's level: approval never widens reach, so an approved write at `read` lands only inside the session's workspace.

However a prompt closes, answered by you or another client, expired, or canceled because the turn ended or nobody was left to answer, the feed says so with `approval.resolved`, carrying the `request_id` and the `outcome`, so a second client showing the same prompt can take it down rather than learn of it from a `404`. While a prompt is parked, the session record counts it in `requests_pending`, on the listing too, which is how a list of sessions says which ones are waiting on a person; the prompts themselves are on the feed, which hands every one still parked to a client that joins. A call a sub-agent makes is asked through its parent's feed and answered on the parent's `responses` route; the event names the sub-agent in `subagent_id`.

### Approvals with JSON responses

A request with `stream: false` never sees an `approval.requested` event, so it is not an answerer. With approvals on, no feed reader attending, and no `answers_prompts_on_return` on the session, every call that would need approval is **refused without asking**; each refused tool appends a `notice` saying so and pointing at `attend=true` and the capability.

**Recommendation:** non-interactive callers (bots, bridges, scripts) should leave approvals off and create sessions at the level they need, so nothing is refused without asking. Use `stream: true`, or attend the feed, with approvals on if you need approval flow; declare `answers_prompts_on_return` on a session a person returns to.

### Elicitations

An MCP server may ask the person something mid-call: a **form**, a JSON Schema of fields to fill, or a **URL** to open and come back from. On a session that declared [`supports_elicitation`](#capabilities), the question parks as an `elicitation.requested` event under the attendance rule approvals follow: someone attending, a streaming request or an `attend=true` reader, or `answers_prompts_on_return`. The event carries the server, the `kind`, the `message`, the `schema` or the `url`, and `expires_in_seconds` with `expires_at`: a minute, or thirty on a session that declared `answers_prompts_on_return` while no request holds the turn, the bound an approval gets there, after which the server is told the person declined; never past the tool call that asked, which `MEKA_MCP_TOOL_TIMEOUT` cuts at ten minutes by default, canceling the prompt with it. The server holds its own request open for as long as it chooses, so the longer wait holds only where the server waits too. The client answers at the same responses route a prompt takes:

```
POST /v1/sessions/{id}/responses/{request_id}
Content-Type: application/json

{"outcome": "accept", "content": {"name": "Ada"}}
```

`accept` carries `content`, an object of the fields the schema named (none for a URL the person opened), and an `accept` without it on a form, or with it on a URL, is a `422`; `decline` says the person would not; `cancel` says they dismissed it. An approval's answer to an elicitation, or an elicitation's to an approval, is a `422` naming the kind the request is, and `content` on anything but `accept` is one too. The feed says how the prompt closed with `elicitation.resolved`, `expired` when its wait ran out and `canceled` when the turn ended, the server withdrew its request or nobody was left to answer, and the session record counts a parked elicitation in `requests_pending` with the approvals. On a session that did not declare the capability, the elicitation is declined without asking, and a `notice` says so.

## Authentication

Every request requires `Authorization: Bearer <token>`, except the two health probes and, when `[serve].docs` is enabled, `/v1/openapi.json`. That one is off by default, so on a default deployment it answers 404 rather than serving anything unauthenticated. A `401` carries `WWW-Authenticate: Bearer realm="meka"`, as RFC 9110 requires. A browser's CORS preflight is the one other exception, and only where [`cors_allowed_origins`](#browser-clients) is set.

### Scopes

Each token carries a set of scopes that control what it can access:

| Scope | Permits |
|-------|---------|
| `sessions:r` | List and search sessions, get details, read the history, the context and its occupancy, turns, export, tools, background tasks, the inbox, the session feed |
| `sessions:w` | Create, modify, delete sessions; submit and cancel turns; enqueue and withdraw inbox items; compact, rewind, import; respond to approval prompts; cancel background tasks. **A session it creates may run at any level `[permissions].enabled` admits, in any directory**, so at the default enabled set, which includes `unrestricted`, this too is shell on the host. The client chooses the directory too, so `workspace` reaches wherever it points, its home included: on a server for clients you do not trust, `read` is the only level that confines one |
| `skills:r` | Read installed skills, including bodies |
| `skills:w` | Create, update, delete skills |
| `memory:r` | Read the memory store |
| `memory:w` | Create, update, delete memories; add, rename and remove projects |
| `schedule:r` | List scheduled jobs. `GET /v1/jobs` is server-wide and returns each job's full `prompt`, so this reads instruction text and not just schedules. A gate's `check` is withheld unless the token also holds `sessions:r` |
| `schedule:w` | Create and cancel scheduled jobs. **A job's `prompt` runs a full turn with tools**, so this is deferred turn execution, not just bookkeeping. A job's optional `gate` runs a shell command or a read-only tool call and additionally requires `sessions:w` (see below) |
| `mcp:r` | Read MCP server status and advertised tools |
| `mcp:w` | Reconnect an MCP server |
| `config:r` | Read the installation: accounts (never a credential), each profile's settings, each MCP server's entry (keys only for `env` and `headers`) |
| `config:w` | Write the installation: add, rename and remove accounts, store their keys, add, change and remove profiles, set the default, add, change, remove and log in MCP servers. **This is the operator's own grant, and it is shell on the host**: an MCP server added with a `command` runs that command as the user `meka serve` runs as, so give it to the client that administers the installation and to nothing else |

Discovery endpoints (`/v1/info`, `/v1/skills`, `/v1/mcp`, `/v1/profiles`, `/v1/projects`) accept any token with at least one read scope. Two deliberately do not: `GET /v1/skills/{name}` needs `skills:r` and `GET /v1/instructions` needs `sessions:r`, because both return instruction *text* rather than a listing.

Scopes are flat: `memory:r` does not imply `memory:w`, and neither implies the other. Operations *on a conversation* stay under `sessions:*`, because the thing being read or changed is one session. The process-wide stores carry their own scopes so a bridge token that runs turns cannot reach them through the API; what its turns' tools reach is bounded by the level and `[permissions]`, not by the token's scopes.

An unrecognized scope logs a warning at startup and grants nothing, so a typo like `sessions:write` is visible rather than silently inert.

> **Note:** `[skills] agent_managed` and `[memory] enabled` govern what the *model* may do on its own initiative. They do not gate these endpoints. A token is the operator acting remotely, equivalent to running `meka skill add` in a shell, so a `skills:w` token writes skills even when `agent_managed = false`.

### Token configuration

Tokens are configured under `[[serve.tokens]]` in your config. Three forms are supported:

```toml
# Inline plaintext, development only (a startup warning is logged)
[[serve.tokens]]
token = "sk_dev_test123"
scopes = ["sessions:r", "sessions:w"]

# Environment variable substitution, recommended for CI/containers
[[serve.tokens]]
token = "${MEKA_BRIDGE_TOKEN}"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]

# File-based, recommended for production (chmod 0600)
[[serve.tokens]]
token_file = "/etc/meka/bridge.token"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]
```

Token comparison uses constant-time equality to prevent timing side-channel attacks. Tokens never appear in logs; only a truncated SHA-256 fingerprint is used for diagnostics.

A change to the tokens applies when the server serves the file: at a write of its own or at `POST /v1/config/reload`. From then on a removed token is refused, and every stream it holds ends, as does one whose token loses a scope the stream needs: `sessions:w` for a streamed turn, both `sessions:r` and `sessions:w` for an attending reader and a reader that keeps its session loaded, `sessions:r` for any other. A request without a stream that is already running, a `stream: false` turn say, finishes and answers.

### Browser clients

A web application served from another origin, a static site or a development server, calls the API directly from the browser once [`[serve].cors_allowed_origins`](../configuration/config-file.md#servecors_allowed_origins) lists its origin, or `*`. The reference page covers the setting; this is what the grant covers.

- **Preflights need no token.** The browser's `OPTIONS` request is answered ahead of authentication and runs nothing: it loads no session, takes no lock and enqueues nothing. The real request that follows needs the same bearer token and scopes as ever.
- **Request headers.** `Authorization`, `Content-Type`, `Idempotency-Key`, `If-Match` and `Last-Event-ID` are granted by name, on top of the headers a browser may always send. `Authorization` has to be named because a wildcard grant never covers it.
- **Methods.** `GET`, `HEAD`, `POST`, `PUT`, `PATCH` and `DELETE`. A method a route does not implement is still a `405`.
- **Response headers.** `ETag`, `Retry-After` and `WWW-Authenticate` are exposed to page script. Errors carry the grant like successes do, so a `401`, a `403`, a `413` or a `429` is a Problem Detail the page can read rather than an opaque failure.
- **No cookies.** `Access-Control-Allow-Credentials` is never sent. Send the bearer header on every request and use `credentials: "omit"`.
- **Streams.** The feed and a streaming turn are granted like any other response and are not buffered. A native `EventSource` cannot send a header, so read SSE with `fetch` and a streaming parser, and send `Last-Event-ID` yourself on a reconnect. Fetch an image or an export the same way and hand the bytes to an object URL.

CORS is the browser's policy on sharing a response, not authorization: an allowed origin still needs a token, and a refused origin only stops the page from reading the answer. It does not make an endpoint reachable, provide TLS, or bypass a browser's local-network permission; a remote deployment is exposed through HTTPS as before. Set the policy in one place: a reverse proxy that adds CORS headers of its own on top of meka's gives the browser two grants, and it refuses both.

## Idempotency

A held JSON turn (`stream: false`, `detach: false`) and a detached request in either encoding support Stripe-style idempotency via the `Idempotency-Key` header:

```bash
curl -X POST http://localhost:8080/v1/sessions/$ID/turns \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: 7f8a9b0c-1234-5678-abcd-ef0123456789" \
  -d '{"message": "deploy to staging"}'
```

If the same key is replayed, the server returns the cached response. If the same key is sent with a different body, it returns `409 Conflict`. A key must be non-empty ASCII of at most 255 characters; anything else is a `422` `invalid-body`. While a request carrying a key is still running, a second request with the same key answers `409` `idempotency` rather than waiting; retry it once the first completes.

Keys are scoped per-token **and per-session**, and expire after 24 hours. The session is part of the scope because an `Idempotency-Key` names *your* unit of work: sending the same key to two sessions is a reasonable thing to do, and it now runs both turns instead of answering the second with the first's transcript.

A held turn that was canceled is not cached, so the retry the cancellation invites can actually run. Neither is a 5xx, for the same reason. Either retry re-executes the turn, and unless the turn was sent with `options.unanswered_message` set to `withdraw` it runs above the message the failed one left behind; see [Resending a failed turn](#resending-a-failed-turn).

A detached request commits its key at admission, with the `202` it was answered. A replay answers that same record for the key's lifetime whatever became of the turn, canceled included, and starts nothing: the client chose not to wait, so the outcome is the turn resource's to report. A detached stream's replay is answered the record too, not a stream.

The cache is bounded per token by both entry count and total bytes; a response too large to keep is not cached, and its retry re-executes.

Idempotency keys are **ignored for a held stream**; such a client reconnects through the [session feed](#the-session-feed) or submits a new turn.

### Which endpoints are safe to retry

`Idempotency-Key` covers held JSON turns and detached requests. For everything else, know what a blind retry does before you configure one:

| Endpoint | Retry-safe | On a duplicate |
|---|---|---|
| `POST /turns` (held JSON, with a key) | yes | cached response returned |
| `POST /turns` (detached, with a key) | yes | the same `202` and turn record; the turn ran once |
| `POST /v1/sessions/{id}/inbox` (with a key) | yes | the earlier item is answered, `replayed: true`; the key is on the row, so this holds across a restart |
| `POST /cancel`, `DELETE /v1/sessions/{id}`, `DELETE /v1/tasks/{task_id}`, `DELETE /v1/sessions/{id}/inbox/{item_id}` | yes | already-done is the same state; a withdrawn item answers **404** on the retry |
| `DELETE /v1/skills/{name}`, `/v1/memories/{name}`, `/v1/projects/{name}`, `/v1/accounts/{name}`, `/v1/profiles/{name}`, `/v1/mcp/{name}`, `/v1/jobs/{job_id}` | yes, but | the resource is gone, so the retry answers **404**. Expected, not a failure; treat it as success if you are retrying blind |
| `PUT /v1/skills/{name}`, `PUT /v1/memories/{name}`, `PUT /v1/instructions`, `PATCH /v1/profiles/{name}` (settings), `PATCH /v1/mcp/{name}`, `POST /v1/accounts/{name}/login` (`api_key`), `POST /v1/mcp/{name}/login` (a secret), `POST /v1/mcp/{name}/logout` | yes | same body writes the same skill file, memory row, instructions, keys or credential; the same secret stored again changes nothing and restarts no server |
| `POST /v1/accounts/{name}/login` (browser), `POST /v1/mcp/{name}/login` (browser), and their completions | yes, but | a second start replaces the first login, whose id then answers **404**; a completion answers again once completed, and **422** once failed |
| `POST /v1/projects`, `POST /v1/accounts`, `POST /v1/profiles`, `POST /v1/mcp` | yes, but | the resource exists, so the retry answers **422** naming it; nothing is added twice |
| `PATCH /v1/projects/{name}`, `PATCH /v1/accounts/{name}`, `PATCH /v1/profiles/{name}` (a `name`) | yes, but | the old name is gone, so the retry answers **404**; the resource is under the new name |
| `POST /compact` | mostly | a second compaction summarizes the summary; fidelity drops, nothing is lost. It takes no `If-Match`, since it addresses no position in the history and a stale view costs a retry nothing it cannot undo |
| `POST /rewind` | yes | a retry names a turn the first attempt already took, so it is refused with `422` and takes nothing; under `If-Match` it is a `412` |
| `POST /fork` | **no** | creates a second copy; `If-Match` does not stop it, since a fork leaves its source as it was |
| `POST /sessions/import` | **no** | creates a second copy of the tree under new ids |
| `POST /v1/jobs` | **no** | creates a second job |

The three marked **no** are operations meant to be driven deliberately. If your HTTP stack retries failed POSTs by default, exclude them, or check the outcome first: `GET /v1/jobs?session=<id>` lists the jobs a session has, and `GET /v1/sessions` the sessions an import or a fork made.

## Error handling

All HTTP error responses use [RFC 9457 Problem Details](https://www.rfc-editor.org/rfc/rfc9457) with `Content-Type: application/problem+json`:

```json
{
  "type": "https://meka.bot/errors/session-not-found",
  "title": "Session not found",
  "status": 404,
  "detail": "session '2f0c9b6e-4d1a-4c0e-9b7f-3a5d8e1c2b4f' not found",
  "instance": "/v1/sessions/s_xyz/turns"
}
```

The `type` URI is the stable, machine-readable error code. Route error handling on `type`, not on `status` or `detail`.

> **A parse failure names the field.** A body that does not parse is a `422` `invalid-body` whose `detail` reads `invalid <endpoint> request body: <what the parser found>`, naming the field at fault: an unknown one, a missing one, or one of the wrong type. Every request type denies unknown fields, so a typo is refused rather than ignored. The names it cites are the wire schema, which the [OpenAPI spec](#endpoint-reference) defines; nothing internal is in them. Validation that runs after parsing, such as a profile that is not configured or a `working_directory` that does not exist, names the field the same way.
>
> A `502` on a turn carries the provider's own response text in a `provider_response` member **when `[serve] relay_provider_errors` is on, which is the default**; a deployment that has turned it off omits the member entirely. It exists because the upstream's error type is the actionable part and a client that cannot see it is left guessing. `detail` stays meka's own sentence either way, so nothing is traded for it, and the relayed text is capped at 4 KiB with the cut marked. The full text goes to the server log regardless.
>
> **`provider_response` is readable at `sessions:r`.** Submitting a turn takes `sessions:w`, but the failure also rides the terminal `turn.failed` event, which `GET /v1/sessions/{id}/feed` replays to any reader. Since an upstream refusal can name the *operator's* account with the provider and its rate-limit posture, set `[serve] relay_provider_errors = false` where read-only tokens go to people who may watch a session but are not entitled to the account behind it.
>
> A turn that failed or was canceled after it began carries `message_withdrawn`, whether the message it was sent is still in the conversation. It is `true` only for a turn that ended before anything from the model reached the conversation and whose opening was to be withdrawn then: one sent with `options.unanswered_message` set to `withdraw`, or on the feed a recurring job's fire or an inbox turn, whose openings are withdrawn the same way. A turn refused before it began, such as the `503` for a required MCP server, never added the message and carries no such member; see [Resending a failed turn](#resending-a-failed-turn).
>
> The `503` a turn gets when a required MCP server is down is not covered by that key and never relays: the server names travel, the connector's reason does not, since it is meka's own subprocess text and has carried a command line and its path. The endpoints under `/v1/mcp` do relay their reason, since a caller naming one server and asking why it will not connect is asking *for* it.
>
> **An installation fault is a `500`, not a `422`.** A `[web]` client meka cannot build from its `proxy` or `ca_cert_file`, or a `base_url` shape a backend refuses, is the operator's to fix and names a path or an endpoint out of their `config.toml`; the body says only "internal server error; consult server logs" and the sentence goes to the log. The common case does not reach a request at all: the web client is built at startup, so a server with a bad `[web]` block fails to start rather than answering turns.

### Error types

| Type | Status | Meaning |
|------|--------|---------|
| `/errors/auth` | 401 | Missing or invalid bearer token |
| `/errors/auth-scope` | 403 | Token lacks the required scope |
| `/errors/session-permission` | 403 | The token is fine; the *session* sits too low. Raise it with `PATCH /v1/sessions/{id}`; a better token will not help |
| `/errors/session-not-found` | 404 | Unknown session id |
| `/errors/not-found` | 404 | Unknown skill, memory, project, account, profile, pending login, MCP server, background task, scheduled job, image blob, inbox item, or turn; also a skill or memory store that is disabled, or a server with `[schedule] enabled = false`, since there is nowhere to write |
| `/errors/session-not-loaded` | 409 | The session exists but is not in memory; submit a turn to load it. Do **not** retry `POST /cancel`: there is no turn to cancel |
| `/errors/session-locked` | 409 | Another meka process holds the session's lock (e.g. two `meka serve` instances sharing one store); wait or restart the other process |
| `/errors/turn-in-flight` | 409 | A turn is already running on this session within *this* process; cancel it via `POST /cancel` first |
| `/errors/turn-canceled` | 409 | Turn was canceled |
| `/errors/turn-mismatch` | 409 | `POST /cancel` named a turn that is not the one in flight; the `turn_id` member names the one that is. Nothing was canceled |
| `/errors/inbox-appended` | 409 | The inbox item is already in the conversation, so only a turn can answer it now; nothing to withdraw |
| `/errors/store-read-only` | 409 | The skill lives under a `[skills] additional_directories` directory; meka reads those but never writes to them, so writing here would shadow the file rather than change it |
| `/errors/instructions-overridden` | 409 | The standing instructions come from `--instructions`, `MEKA_INSTRUCTIONS`, `MEKA_INSTRUCTIONS_FILE` or the `instructions/` directory, which stand above `instructions.md`, so `PUT /v1/instructions` would change nothing a session sees; `detail` names the source |
| `/errors/account-in-use` | 409 | A session or sub-agent running on this server may run on the account named for a rename, through its profile or one it falls back to, or another meka is refreshing the account's credential, which refreshes under its name; delete the sessions or wait for them to be evicted, or retry shortly |
| `/errors/subagent-not-running` | 409 | The id names a sub-agent this process is not running, so there is no feed to read; its feed exists while its parent runs it, and `last_turn` on its record says how its last run ended |
| `/errors/session-not-drivable` | 422 | The id names a sub-agent's conversation, which only its parent drives. Reading it is unaffected; the message names the parent and what to do there: `agent_followup` for a turn or a fork, `POST /v1/sessions/{parent}/responses/{request_id}` for an approval, the parent itself for a scheduled job, and the parent's feed for a sub-agent's `attend`. **Do not retry with a corrected payload**: no body addressed at this id is accepted |
| `/errors/request-not-found` | 404 | Unknown or expired `request_id` |
| `/errors/idempotency` | 409/429 | Key conflict (body mismatch: 409; cache cap: 429) |
| `/errors/invalid-body` | 400/422 | Request body validation failed (422), or a path/query parameter the router rejected (400) |
| `/errors/precondition-failed` | 412 | `If-Match` on a rewind or a fork names a conversation that has since changed; `position` carries the current one. Read `GET /history` or `GET /context/messages` again and decide again |
| `/errors/request-too-large` | 422 | meka refused to send the turn: the conversation is still over the profile's `max_request_bytes` after redacting older tool-result images. meka's own ceiling, so no provider judged it and no `provider_response` rides along; `detail` names the size, the limit and the remedy, `/compact`. **Do not retry unchanged**: `POST /compact` first |
| `/errors/payload-too-large` | 413 | Body exceeds `max_body_bytes`, meka's limit on the HTTP request itself. Unrelated to `request-too-large`, which is about what meka may send onward |
| `/errors/concurrency-limit` | 429 | Process-wide turn limit reached (`Retry-After` header included) |
| `/errors/sse-lag` | 500 | SSE consumer fell behind; stream terminated (see [Falling behind](#falling-behind)) |
| `/errors/stream-detached` | 500 | SSE-only. A re-attached stream ended with no recorded outcome because the turn's task died; read `GET /history` for what completed |
| `/errors/provider` | 502 | An upstream call failed for a reason meka could not classify as transient. Usually permanent (a revoked credential, a `base_url` that is not the API), but it is a catch-all, so treat it as "no reason to expect a retry to help" rather than "a retry cannot help" |
| `/errors/provider-unavailable` | 502 | The upstream failed in a way meka's classifier had already labeled transient. **Worth one backed-off resend.** Carries a `Retry-After` when the upstream gave one, which most of the time it did not |
| `/errors/provider-exhausted` | 502 | The upstream refused the request for quota or billing, by a signal it documents: a 402, a quota code on a 429, or the Claude subscription's exhausted usage window. **Nothing changes until the reset**, which rides as `Retry-After` when the upstream stated one. A profile with [`fallbacks`](../configuration/config-file.md#fallbacks) hands the turn to the next one instead of answering this |
| `/errors/context-overflow` | 502 | The conversation exceeds the model's context window, and auto-compaction was off, already spent, or could not shorten it enough. **Do not retry unchanged**; `POST /compact` or send less first. Carries `provider_response` like the two above, since the upstream is what refused it |
| `/errors/mcp-unavailable` | 503 | An MCP server marked `required` was not connected, so the turn was refused before reaching the provider. The `servers` extension names them; each one's reason is in the server log |
| `/errors/shutting-down` | 503 | The server has begun shutting down and takes on nothing that would outlive it, such as a browser login begun or committing now |
| `/errors/internal` | 500 | Unhandled server error |

Streaming turns that fail mid-stream emit a `turn.failed` SSE event with the same error shape, then close the connection.

> The three 502s are the ones worth branching on. `/errors/provider-unavailable` is the positive signal: meka's classifier recognized the failure as transient, which covers an overload, a 5xx, a dropped connection and a stalled stream. Resend it after a pause; [Resending a failed turn](#resending-a-failed-turn) says what the failed turn leaves behind. `/errors/context-overflow` is the flat refusal: the request no longer fits and will not fit next time either, so retrying it unchanged loops until your client gives up; shorten the conversation with `POST /v1/sessions/{id}/compact` or send less.
>
> **`/errors/provider` is the absence of the first signal, not the opposite of it.** It is a catch-all covering everything meka could not place, so a revoked credential lands there and so does a 408, a truncated response body, and any mid-stream error type meka does not yet recognize. Most of the time it is permanent and worth surfacing to a human rather than retrying, but do not build a client that will *never* retry it: one unhurried resend is reasonable, an unbounded loop is not.
>
> **Branch on `type`, not on `Retry-After`.** A `Retry-After` is present only when the upstream volunteered one in delta-seconds form, which most transient failures do not: a dropped connection never produced a response to carry a header, a mid-stream `overloaded_error` has no headers at all, and an upstream answering with an HTTP date sends none meka can read. Treating its absence as "permanent" discards turns a second attempt would have completed, which is the reason these two types exist separately.
>
> Neither provider type says how many attempts meka made first. It declines to retry at all once any output has reached the stream or its retry budget is spent, and a canceled turn abandons the sequence wherever it stands, so one of these can reach you after three attempts or after none. `/errors/provider-unavailable` claims a failure class, not that your next attempt will succeed.
>
> A `Retry-After` on a `/errors/provider-unavailable` response is the upstream's own, relayed up to an hour. Honor it in preference to your own backoff. The other two never carry one.

### Resending a failed turn

A failed turn keeps the message you sent, in the conversation and on disk. That is the screen's
behavior too: the person who typed the prompt can see it and expects the agent to have it, and a
turn that got as far as a partial answer or a tool call has work behind it that refers to the
message. For a client that answers a `502` by resending the same message it is the wrong default,
because the resend appends a second copy of the message after the first, and the model is then shown
the same request twice with nothing between them for the life of the session.

Say so instead. A turn sent with `options.unanswered_message` set to `withdraw` takes its message
back when the turn ends, failed or canceled, before anything from the model reached the
conversation, so the resend is the only copy. A turn that got a partial answer or a tool call into
the conversation keeps its message either way, so a resend after one is a new turn rather than a
replay. Send the option on every turn you would resend; it is per turn, not per session, and it
never touches a background outcome that was riding on the message, whose row is already spent.

The response says which happened. A turn that failed or was canceled after it began carries
`message_withdrawn`, in the Problem Detail body of a blocking turn and on the `turn.failed` and
`turn.canceled` events of a streaming one. Branch on it rather than on what you saw arrive: thinking
and a half-composed tool call both look like output on the stream, and neither reaches the
conversation, while a completed reply of nothing but thinking does and is invisible to a blocking
client. `true` means the conversation no longer holds the message and your resend will be the only
copy. `false` means it does, and a resend appends a second one after whatever the turn produced. A
turn refused before it began, such as the `503` for a required MCP server or any `4xx`, never added
the message and carries no `message_withdrawn`; a resend is the first copy.

## Discovery endpoints

These endpoints help clients inspect the server's capabilities at runtime.

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /v1/health/live` | None | Liveness probe: 200 if the process is up |
| `GET /v1/health/ready` | None | Readiness probe: 200 if the store is healthy, at least one profile is configured, and no `required` MCP server has failed. A failed *optional* server doesn't affect readiness, since it can't stop a turn either. Returns `status`, `session_db`, `profile_configured`, and `mcp_servers_healthy` (boolean, no server names). **`profile_configured` means a profile exists in `config.toml`, not that it has a usable credential**: a profile's credential is checked when a session first needs it, so a server can be ready and still answer 422 to `POST /v1/sessions`. |
| `GET /v1/projects` | Any read scope | Projects, as `{"projects": [...]}`: every project, each with `name`, `created_at` (RFC 3339), `memories` (in its tier) and `sessions` (root sessions on it). `POST /v1/projects` adds one, `PATCH /v1/projects/{name}` renames one and `DELETE /v1/projects/{name}` removes one, under `memory:w`; see [Projects](projects.md#defining-a-project) |
| `GET /v1/profiles` | Any read scope | Configured profiles, as `{"profiles": [...]}`. Each carries `name`, `account`, `backend` (omitted when the profile names an account that is not configured), `model` (omitted when the profile names none), `fallbacks`, the profiles a turn moves to when this one cannot serve it (omitted when it names none), `vision`, whether a session on it accepts [image attachments](#image-attachments), and `default: true` on the one a session gets when it names none. `POST`, `PATCH` and `DELETE` on `/v1/profiles` manage them under `config:w`; see [Managing the installation](#managing-the-installation) |
| `GET /v1/info` | Any read scope | Server version and permission surface, and `scopes`, the ones the calling token holds, so a client can show only the controls it may use. Where the installation is, and whether it awaits a reload or a restart, is `GET /v1/config`'s, under `config:r` (see [Managing the installation](#managing-the-installation)). Carries no profile, model or capability of one: `GET /v1/profiles` reports those per profile and marks the default with `default` |
| `GET /v1/skills` | Any read scope | Installed skills |
| `GET /v1/mcp` | Any read scope | MCP server connection status, with the protocol version a connected one speaks, and `orphaned_credentials` |
| `GET /v1/openapi.json` | None, and off unless `[serve].docs` is set | OpenAPI 3 spec |

## Managing the installation

What `meka account` and `meka profile` do in a shell, `/v1/accounts` and `/v1/profiles` do over HTTP, under `config:r` and `config:w`, so a client that administers an installation (a desktop app wrapping `meka serve`, a web UI on a server nobody logs in to) can take it from an empty `config.toml` to a running profile without a shell. Every write goes through the same editor the CLI uses: `config.toml` keeps its comments and its order, the same refusals apply in the same words, and a credential goes to the database and never to the file.

An **account** is added with `POST /v1/accounts`, `{"name", "backend"}` and whatever of `base_url`, `oauth_token_url`, `client_id`, `interleaved_thinking`, `prompt_cache_key`, `stream_usage` and `max_tool_name_bytes` the backend reads; a setting the backend never reads is a `422` rather than written and ignored. An API-key backend takes its key as `api_key` in the same body, and the key is required there; a subscription backend takes none, is added unauthenticated, and is logged in afterwards. `POST /v1/accounts/{name}/login` with `{"api_key"}` rotates the key of an API-key account. On a subscription account it takes no body and begins the browser login `meka account login` runs at a terminal, answering `202` with `authorization_url`, the page the person opens, and a `login_id`; what the page gives is posted as `code` to `POST /v1/accounts/{name}/login/{login_id}`, which exchanges it, stores the credential and answers the account. `completes_with` says what that is: `code` on `claude-subscription`, where Anthropic's page shows a code to copy, and `callback_url` on `chatgpt-subscription`, where the browser is sent to `http://localhost:1455/auth/callback`. For the latter the server holds that loopback listener itself while the login is pending, when the port is free (`listening: true`), so a browser on the server's own machine completes the login with nothing posted; a client whose browser is elsewhere posts the URL the browser landed on. `GET /v1/accounts/{name}/login/{login_id}` says whether the login is `pending`, `completed` or `failed` and why. A login lapses after ten minutes, a wrong paste leaves it pending, an exchange the authorization server rejects ends it (the code is spent, so begin another), and beginning a login on an account that has one pending replaces it. A login stores its credential only while the account states what it began on: another write meanwhile leaves it pending, a rename or a removal of the account ends it, and an account moved by hand to another backend, client or token endpoint answers the completion with a `422` and stores nothing. A token endpoint that cannot be reached, or answers with something that is not a token, ends the login with a `502` and the reason in the meka log. Once the exchange has a credential, the login is the commit's to finish, so its completion waits for the commit even past `expires_at`, and a login begun while the server shuts down is a `503` `shutting-down`. `PATCH /v1/accounts/{name}` with `{"name"}` renames the account, moving its credential and every profile on it, and is the only change an account takes: its other settings are what a login was made against, so another endpoint or backend is another account. A subscription account holding a credential is not renamed while a session open on the server, or a sub-agent running there, may run on it, through its profile or one that profile falls back to, or while another meka refreshes its credential (`409` `account-in-use`): a running provider refreshes the credential under the old name. Rename it once those sessions have closed. An account is not added or renamed onto a name a removed account's credential is still stored under, which `orphaned_credentials` lists: adopted, a credential minted for another backend or endpoint would be presented, so it is a `422` naming `DELETE /v1/accounts/{name}`, which clears it. A name with whitespace at either end is a `422` on every door that adds or renames an account or a profile. `DELETE /v1/accounts/{name}` removes the account and the credential stored under its name, or a credential alone when no account claims it, and is a `422` while a profile names the account. `GET /v1/accounts` lists them with `authenticated` (`yes`, `no` or `unreadable`) and, when there are any, `orphaned_credentials`, the names a credential is stored under that no account claims, the file's included, so an account added in a shell and not yet served is not one. `GET /v1/accounts/{name}` answers one; each carries every setting the account states, `base_url`, `oauth_token_url`, `client_id`, `interleaved_thinking`, `prompt_cache_key`, `stream_usage` and `max_tool_name_bytes`, beside `backend`.

A **profile** is added with `POST /v1/profiles`, `{"name", "account", "model"}` and any of the [profile fields](../configuration/config-file.md#profile-fields); a field left out is left out of the profile, so its documented default applies, and `"default": true` makes it `default_profile`. The first profile needs no `default`: a sole profile is the default by the selection rule, and `default` says so. The second profile added names the first as `default_profile` unless it takes `default` itself, so the default stays where it was rather than lapsing with nothing in the answer to say so. `GET /v1/profiles/{name}` answers every setting the profile states. `PATCH /v1/profiles/{name}` changes settings in one write: a key left out keeps its value, `null` removes the key (an empty string does the same on a string key), a value sets it, and `"default": true` makes the profile the default in the same write; `"default": false` is refused, since the default is unset by removing the profile or choosing another. [`parameters`](../configuration/config-file.md#parameters) is one value: an object replaces the whole table and `null` removes it, so a field is dropped by sending the table without it, and a `null` inside the object, which TOML has no word for, is a `422`. A `{"name"}` on the `PATCH` renames the profile, moving every session recorded on it and the name every resident session and running sub-agent holds, so a worker spawned mid-turn records the new one, and is sent on its own; a rename made from a shell reaches the rows alone, and a resident session takes it at its next turn. The profile `meka serve --profile` named is neither renamed nor removed while the server runs, and no other is made the default (`422`): it is the server's default until it restarts. `fallbacks` are read the same on an add as on a change, each name trimmed and a blank one dropped. `DELETE /v1/profiles/{name}` removes it; sessions recorded on the name keep it and refuse to resume until it is configured again or they are moved, a profile another's `fallbacks` names is a `422` until that list drops it, and the default is a `422` while two or more profiles would remain, naming `PATCH /v1/profiles/{name}` with `"default": true` as the way to move it first; with one profile left the key is dropped, and that profile is the default by the rule. A setting the profile's backend never sends, a `model` that is blank, a fallback chain the profile could not run on, and an account that is not configured are each a `422` carrying the sentence `meka profile` would print. A `config.toml` that fails to parse, a key added by hand while the server runs say, or that holds anything else startup would refuse, an MCP server entry, a profile that could not run, a fallback chain naming no profile, a `[serve]` token whose variable is unset or `[web]` settings no client can be built from, refuses every write of the installation and every reload with a `422` naming the defect until the file is repaired in a shell, since the whole file is served after each write; a parse error is named by its line and the parser's message, never by the line's content, which may hold a value no listing shows. The file each write serves is held to all of it too, so a shell's edit landing between a write and its serving is never served unchecked: one that leaves the file unservable answers the write with a `422` saying the change is written and not served. A write that fails partway undoes what it wrote: an add takes its entry back out and a change puts the entry back as it was. A removal that fails after clearing part of what it removes is finished by removing again, which clears what is left, an entry with nothing to present or a credential no entry claims.

An **MCP server** is added with `POST /v1/mcp`, whose body is the `[[mcp.servers]]` entry as [the reference](../configuration/config-file.md#mcpservers) documents it, `name` and `transport` included, with `auth_token` or `client_secret` beside it for a secret the entry presents, which goes to the store; a key that does not fit the transport (`headers_helper` among the http ones), a name in use or reserved, a secret the entry cannot present, and a `client_credentials` block with no `client_secret` to present are each a `422`. The server joins the running manager at once and connects in the background: the answer's `state` is `pending`, `GET /v1/mcp` says when it is `connected`, and its tools reach every session as they are listed. `GET /v1/mcp/{name}` answers the entry as `meka mcp get` prints it: every key as written, the `auth` block included (keys only for `env` and `headers`, which may hold a secret), `required` and `deferral` as the server runs under them with `inherited` naming those the entry leaves to `[mcp]`, the kinds of credential stored and presented to its `url` as `credentials`, each kind it holds for another endpoint under `credentials_for_another_endpoint` with the origin it was stored for, and the connection state. `PATCH /v1/mcp/{name}` changes keys in place, each left out to keep, `null` to remove, a value to set, `auth_token` and `client_secret` included; `env`, `headers` and `tool_permissions` merge key by key, a key set to `null` removing that one, since the detail shows the keys of the first two and never their values. It restarts the server when a key its connection is made from or a credential it stores changed; the same secret again is no change. A credential is bound to the endpoint it was stored for and presented there alone: an OAuth login's tokens to that exact `url`, a bearer or a client secret anywhere on its scheme, host and port. A `PATCH` that moves `url` off the endpoint a login's tokens were issued for, or changes `auth`, clears those tokens, revoking them at the provider once the change commits, and the server needs a login again; a bearer or a client secret on another origin stays stored and unpresented until a `PATCH` sends it again or a logout clears it. A change to its tool policy alone (`permission`, `tool_permissions`, `trust_read_only_hint`, `allowed_tools`, `disabled_tools`, `eager_load_tools`, `deferral`, `required`) applies on the connection it holds: the server lists its tools again under the new policy and every session takes the new set, so a stdio server keeps whatever state its process holds. A change to nothing leaves the server alone. `{"disabled": true}` stops it and `{"disabled": false}` starts it, and either way the file keeps its comments. `name` and `transport` are not changed: another transport is another server. `DELETE /v1/mcp/{name}` removes the entry, closes the connection or ends a connect still in flight, drops the server's tools from every session and clears its credentials, revoking an OAuth grant at the provider once the removal commits; a name with credentials and no entry, which `GET /v1/mcp` lists under `orphaned_credentials` (a name neither the file nor the running manager holds), is cleared the same way, and `POST /v1/mcp` with that name is a `422` naming the route until it is. `POST /v1/mcp/{name}/login` with `{"auth_token"}` or `{"client_secret"}` stores the secret as `meka mcp login --auth-token-stdin` and `--client-secret-stdin` do, and with no body begins the OAuth login `meka mcp login` runs at a terminal, answered as an account's browser login is: `202` with the URL and a `login_id`, the loopback listener held by the server, a browser elsewhere answered by posting its callback URL to `POST /v1/mcp/{name}/login/{login_id}`, and `GET /v1/mcp/{name}/login/{login_id}` for where it stands; a paste with no code in it, or carrying another login's state, is a `422` that leaves the login pending, as an account's is; the flow runs against a store of its own, so the bundle stored for the server, and the running server's use of it, are untouched until the login commits. It commits only while the file states the connection it began on: another write meanwhile leaves it pending, removing the server ends it, and a change to the server's endpoint or `auth` block answers the completion with a `422` that stores nothing. Once the credential is stored the server is restarted from its entry as the file then states it, so an `auth` block the login wrote is in force; the restart connects in the background, so the completion's answer carries `state: pending` until it has. An MCP login lapses after two and a half minutes, the flow's own bound rather than an account's ten; `expires_at` says when. A login an endpoint cannot begin, one with nothing to discover, is a `502` with the reason in the meka log, and one on a `disabled` entry, or while a bearer is stored for the server, is a `422`, the latter naming `POST /v1/mcp/{name}/logout`. `POST /v1/mcp/{name}/logout` clears every credential stored for the server and restarts it from its entry, so it runs without them, revoking the OAuth grant it held once the logout commits; a server that held none is left as it is. A `client_credentials` server needs no browser: `POST /v1/mcp/{name}/login` with no body connects it with its stored secret and answers `200` with the server, as storing a secret does. A server that holds no OAuth credentials for its `url` is `failed` with an `error` naming this route, and so is one with no `auth` block and no bearer that answers with a challenge, since the login, which assumes `oauth`, is what connects it.

The **standing instructions** are written with `PUT /v1/instructions`, `{"content": "..."}`, which lands in `instructions.md` in the config directory; a blank `content` removes the file. `GET /v1/instructions` says whether that file is what this server reads, `writable`, and, to a token that holds `config:r` or `config:w`, names it in `path` and where the instructions came from in `source`; when `--instructions`, `MEKA_INSTRUCTIONS`, `MEKA_INSTRUCTIONS_FILE` or an `instructions/` directory with files in it stands above it, a write would change nothing a session sees and is refused with `409` naming the source. The sessions that open after the write run under it; one already open keeps what it opened under.

`GET /v1/config`, under `config:r`, names `config_path` and `data_dir`, for a client that has to find the installation, and carries `config_loaded_at`, when this process last read `config.toml` whole, beside `config_modified_at`, when the file was last written, `config_changed`, whether the file has changed since that read, and `restart_required`, the sections whose content differs from the file the server started on and that only a restart applies, empty while the file fails to parse: `permissions`, `shell`, `[mcp]`'s own keys as `mcp`, and the others [When edits take effect](../configuration/overview.md#when-edits-take-effect) lists. A write of the server's own reads the file again, so only an edit made in a shell, which `POST /v1/config/reload` applies, sets `config_changed`; the `device_id` meka writes into a `claude-subscription` account that states none counts as read.

A write **applies at once**: the accounts and profiles every session on the server resolves against are replaced with what the file now holds, so the profile a client adds is the one its next `POST /v1/sessions` runs on, and the key it rotates is the one the next session presents. A session already resident keeps the provider it was built with until it is evicted or repinned, as it does across `meka account login` from a shell. An MCP server added, changed or removed is added to, restarted on, reconfigured on or taken out of the running manager, so its tools come and go from every session without a restart. An edit made by `meka profile add`, `meka mcp add` or by hand while the server runs is applied by `POST /v1/config/reload`, which reads the file again and serves it the way a write of the server's own is served: the catalog of accounts and profiles is replaced, with the `[thinking]` budget and `[session].context_window` a profile stating none falls back to, an MCP server the file now names is started, one whose connection changed is restarted on it, one whose tool policy alone changed takes it in place, one the file no longer names is taken out, and the standing instructions are read again from wherever this server reads them, the last text staying in force when they cannot be. A credential stored from a shell, by `meka mcp login` say, restarts its server too, since the store counts every write of a server's credentials. The answer names the servers it started, restarted, reconfigured and removed, and `config_loaded_at` moves to the read. Every other section of the file is read at startup alone, as [When edits take effect](../configuration/overview.md#when-edits-take-effect) says, and the answer's `restart_required` names those the file has changed since the server started. Every write of the server's own ends with the same read of the whole file, so an edit made in a shell before it, to any of the four, is served by that write; one made after it waits for the next write, a reload or a restart.

## Session lifecycle

### Idle timeout and GC

A background garbage collector scans in-memory sessions and evicts those that have been idle longer than `idle_timeout`; `"0s"` disables it, and `gc_scan_interval = "0s"` is refused:

```toml
[serve]
idle_timeout = "24h"
gc_scan_interval = "5m"
```

Eviction drops the in-memory state (agent runtime, conversation buffer, cancellation tokens) but **keeps the SQLite row**. A later request with the same session id transparently re-attaches and continues the conversation.

To also remove the row on eviction:

```toml
[serve]
delete_on_idle = true
```

A session is never evicted while a turn is in flight, while a scheduled fire or a compaction holds its runtime, or while one of its background tasks is still running.

### Graceful shutdown

`meka serve` handles `SIGTERM` / `SIGINT` with a controlled drain:

1. Stop accepting new connections.
2. Cancel all in-flight turns (same mechanism as `POST /cancel`).
3. Emit `turn.canceled` with `reason: "server_shutdown"` on open SSE streams.
4. Wait up to `shutdown_drain_timeout` for every turn to finish unwinding, including scheduled
   fires, background-outcome deliveries, and turns whose client has already disconnected.
   Canceling a turn is not the same as waiting for one: what follows the cancellation is the
   commit of the partial reply and of whatever the round already produced.
5. Exit `0`. A drain that hits the timeout instead logs a warning, abandons what is still
   running, and exits `1`, so a supervisor can tell the two apart.

```toml
[serve]
shutdown_drain_timeout = "30s"
```

## Concurrency

- **Per session:** one turn at a time. A second `POST /turns` returns 409; a message that should not wait for the session to be free goes through the [inbox](#the-inbox), which a running turn reads at its next round boundary.
- **Across sessions:** fully concurrent. Multiple sessions can run turns in parallel.
- **Process-wide cap (optional):** set `max_concurrent_turns` to limit total in-flight turns, autonomous ones (inbox, scheduled, background outcomes) included. A client turn past the cap returns 429 with a `Retry-After` header; an autonomous one waits for a free slot.

## Configuration

All settings live under `[serve]` in your `config.toml`. See the [`[serve]` section](../configuration/config-file.md#serve) of the config file reference for the full field list.

Minimal example:

```toml
[serve]
bind = "127.0.0.1:8080"

[[serve.tokens]]
token = "${MEKA_API_TOKEN}"
scopes = ["sessions:r", "sessions:w"]
```

Full example:

```toml
[serve]
bind = "0.0.0.0:8080"
cors_allowed_origins = ["https://owner.github.io"]   # a browser UI's origin; omit for none
max_body_bytes = 10485760           # 10 MiB (default)
max_concurrent_turns = 20
idle_timeout = "24h"
gc_scan_interval = "5m"
delete_on_idle = false
shutdown_drain_timeout = "30s"

# Bridge token, env var substitution
[[serve.tokens]]
token = "${BRIDGE_TOKEN}"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]

# Admin token, file-based
[[serve.tokens]]
token_file = "/etc/meka/admin.token"
description = "operator debugging"
scopes = ["sessions:r", "sessions:w", "mcp:r", "skills:r"]
```

## Client recipes

### Telegram bridge (Python)

A bridge lives with the session, so it submits through the inbox and watches the feed rather than holding a turn open per message. The agent's replies go out through whatever tool the bridge exposes to it; the feed tells the bridge when each message was read.

```python
import httpx

MEKA_URL = "http://localhost:8080"
MEKA_TOKEN = os.environ["MEKA_TOKEN"]
HEADERS = {"Authorization": f"Bearer {MEKA_TOKEN}"}

async def handle_message(chat_id: str, message_id: str, text: str):
    session_id = await get_or_create_session(chat_id)
    resp = await httpx.AsyncClient().post(
        f"{MEKA_URL}/v1/sessions/{session_id}/inbox",
        headers={**HEADERS, "Idempotency-Key": f"{chat_id}:{message_id}"},
        json={"message": text, "class": "steer", "source": "telegram"},
    )
    resp.raise_for_status()
    return resp.json()["item_id"]  # 202: durable, read at the next boundary

async def follow(session_id: str):
    async with httpx.AsyncClient(timeout=None).stream(
        "GET", f"{MEKA_URL}/v1/sessions/{session_id}/feed", headers=HEADERS
    ) as feed:
        async for event in parse_sse(feed):
            if event.name == "inbox.delivered":
                mark_read(event.data["item_ids"])
            elif event.name == "inbox.failed":
                tell_chat(event.data["item_id"], event.data["reason"])
```

### Web UI (TypeScript, streaming)

A UI that renders the whole session reads its [history](#history), subscribes to the feed once and files events by `turn_id`, so it also shows the turns it did not start: a scheduled fire, a background task reporting, a message the user typed while the agent was working and the agent answering it in place. At each terminal it reads the history after the last position it holds, which replaces what it drew live with what was stored. A UI that also keeps a list of sessions holds [the server feed](#the-server-feed) beside it and replaces a row whenever a `session.*` event names it. The feed is read with `fetch` and an SSE parser rather than a native `EventSource`, which cannot send the bearer header; a UI served from another origin also needs [`cors_allowed_origins`](#browser-clients) to name it.

```typescript
const headers = { Authorization: `Bearer ${token}` };
const session = `${MEKA_URL}/v1/sessions/${sessionId}`;

const page = await (await fetch(`${session}/history`, { headers })).json();
let held = page.position; // older pages: `?before=${page.next_before}` while it is present
render(page.items);

const terminals = ["turn.finished", "turn.failed", "turn.canceled"];
const feed = await fetch(`${session}/feed`, { headers });
for await (const event of parseSse(feed.body)) { // any SSE parser over a ReadableStream
  const data = JSON.parse(event.data);
  if (event.event === "turn.started") openTurn(data);
  if (event.event === "assistant_text.delta") append(data);
  if (terminals.includes(event.event) && data.position !== undefined) {
    const after = await (await fetch(`${session}/history?after=${held}`, { headers })).json();
    closeTurn(data, after.items); // what the turn wrote, as stored
    held = data.position;
  }
}

async function send(input: string) {
  await fetch(`${MEKA_URL}/v1/sessions/${sessionId}/inbox`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({ message: input, class: "steer" }),
  });
}
```

A UI that wants one turn per request, with the response scoped to it, keeps `POST /turns` with `stream: true`; that stream still closes at its own terminal.

### Shell script

```bash
#!/usr/bin/env bash
set -euo pipefail

TOKEN="sk_..."
BASE="http://localhost:8080"

# Create a session
SESSION=$(curl -sf -X POST "$BASE/v1/sessions" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"working_directory\": \"$(pwd)\"}" | jq -r .id)

# Run a turn
RESULT=$(curl -sf -X POST "$BASE/v1/sessions/$SESSION/turns" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "summarize this project"}')

# The reply: the text blocks of the assistant's messages, joined.
echo "$RESULT" | jq -r '[.messages[] | select(.role == "assistant") | .content[] | select(.type == "text") | .text] | join("")'

# Clean up
curl -sf -X DELETE "$BASE/v1/sessions/$SESSION" \
  -H "Authorization: Bearer $TOKEN"
```

## Scheduled jobs

`meka serve` is the durable host for [scheduled wakeups](./scheduling.md). It fires every job in the store, reviving evicted sessions on demand, so jobs keep running whether or not a client is connected and survive a restart of the server.

An agent-initiated turn has no HTTP request to respond to, so its output is persisted to the session like any other turn. Read it back with `GET /v1/sessions/{id}/history`, or follow it on the [session feed](#the-session-feed).

```
GET    /v1/jobs            Every scheduled job, or one session's with ?session=<id>
POST   /v1/jobs            Plant a job on a session
DELETE /v1/jobs/{job_id}   Cancel a job by its full id
```

`POST /v1/jobs` plants a job on the session its `session_id` names, and answers `201` with the
job; an id no session has is a `404`. Scheduling must be enabled on the server
(`[schedule] enabled`), or the request is a `404` `not-found`: there is nowhere for the job to go,
the same answer a disabled skill or memory store gives. Listing and canceling stay open, so jobs
left from before the flag was flipped can still be cleared out. `GET /v1/jobs?session=<id>` answers
an empty list for a session with no jobs, or an id no session has.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `session_id` | string | *(required)* | The session the job fires in |
| `prompt` | string | *(required)* | What the agent is asked to do when the job fires; must not be blank |
| `at` | string | | One-shot: an RFC 3339 instant, or a duration from now (`"20m"`, `"1h 30m"`) |
| `every` | string | | Recurring interval (`"30m"`, `"6h"`) |
| `cron` | string | | 5-field cron pattern, read in `timezone` |
| `timezone` | string | the host's | IANA time zone a `cron` pattern is read in; refused beside `at` or `every` |
| `gate` | object | | Guard the job on a probe; see below |

Exactly one of `at`, `every` and `cron` is required. A `gate` is `{"check": ..., "when": ...}`:
`check` is `{"command": "..."}` for a shell command or `{"tool": "...", "arguments": {...}}` for a
read-only tool call, and `when` is `"changed"` (the default), `"succeeded"`, `{"matches": "<regex>"}`
or `{"at": "<json pointer>", "is": "not_empty" | "empty" | "changed"}`. What a gate requires of the
token and the session is under [Endpoint reference](#endpoint-reference).

## Reverse proxy setup

For production deployments behind nginx:

```nginx
location /v1/ {
    proxy_pass http://127.0.0.1:8080;
    proxy_buffering off;
    proxy_cache off;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_read_timeout 600s;
}
```

Key points:
- **Disable buffering**: SSE events must not be buffered. Every SSE response carries `X-Accel-Buffering: no` and `Cache-Control: no-cache, no-transform`, which switch nginx's buffering off per response; the `proxy_buffering off` above covers proxies that do not honor the header.
- **Extend read timeout**: turns can take minutes; the default 60s is too short.
- **Do not compress**: gzip/brotli on SSE responses swallow events. Exclude the `/turns`, `/feed` and `/v1/feed` routes from compression middleware.

## Endpoint reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/health/live` | None | Liveness probe |
| GET | `/v1/health/ready` | None | Readiness probe |
| GET | `/v1/info` | read | Server version, permission surface and the caller's scopes |
| GET | `/v1/skills` | read | Installed skills |
| GET | `/v1/mcp` | read | `servers`, each with `state` and `protocol_version` while `connected`; `orphaned_credentials` when a credential is stored under a name no entry claims |
| POST | `/v1/sessions` | `sessions:w` | Create session |
| GET | `/v1/sessions` | `sessions:r` | List sessions, or with `?q=<words>` the sessions whose conversations hold the words, best first |
| GET | `/v1/sessions/{id}` | `sessions:r` | Get session |
| PATCH | `/v1/sessions/{id}` | `sessions:w` | Update session |
| DELETE | `/v1/sessions/{id}` | `sessions:w` | Delete session |
| POST | `/v1/sessions/{id}/fork` | `sessions:w` | Fork session, or with `turn_id` branch it before a turn; `If-Match` makes it conditional |
| GET | `/v1/sessions/{id}/history` | `sessions:r` | A page of what the session said, paged by `before` or `after`, under the conversation's `ETag` |
| GET | `/v1/sessions/{id}/context/messages` | `sessions:r` | Every message the model sees, under the conversation's `ETag` |
| GET | `/v1/sessions/{id}/blobs/{hash}` | `sessions:r` | Image bytes behind a content block |
| POST | `/v1/sessions/{id}/turns` | `sessions:w` | Submit turn |
| POST | `/v1/sessions/{id}/cancel` | `sessions:w` | Cancel turn (optionally one named `turn_id`) |
| POST | `/v1/sessions/{id}/inbox` | `sessions:w` | Enqueue a message: `steer` reaches the running turn, `interrupt` cuts its answer, `followup` waits |
| GET | `/v1/sessions/{id}/inbox` | `sessions:r` | Inbox items the model has not been shown |
| DELETE | `/v1/sessions/{id}/inbox/{item_id}` | `sessions:w` | Withdraw an item still waiting |
| POST | `/v1/sessions/{id}/responses/{request_id}` | `sessions:w` | Answer a parked prompt: an approval (`allow`, `deny`, `allow_always`, `deny_always`) or an elicitation (`accept` with `content`, `decline`, `cancel`) |
| GET | `/v1/sessions/{id}/feed` | `sessions:r` | The session's event feed, across turns; `sessions:w` to load a session this process has not, and `?attend=true` (needs `sessions:w`) to be asked to approve gated calls. On a sub-agent's id, its own feed while its parent runs it, read-only |
| GET | `/v1/feed` | `sessions:r` | The server feed: every change to a session record, across sessions |
| POST | `/v1/sessions/{id}/compact` | `sessions:w` | Summarize the conversation now |
| GET | `/v1/sessions/{id}/context` | `sessions:r` | Context occupancy |
| POST | `/v1/sessions/{id}/rewind` | `sessions:w` | Take the conversation back to before a turn; `If-Match` makes it conditional |
| GET | `/v1/sessions/{id}/export` | `sessions:r` | The session's archive, as `POST /v1/sessions/import` takes it |
| GET | `/v1/sessions/{id}/turns` | `sessions:r` | The turns that began, newest first, paginated by `before` |
| GET | `/v1/sessions/{id}/turns/{turn_id}` | `sessions:r` | One turn, with its ending once it has one |
| POST | `/v1/sessions/import` | `sessions:w` | Recreate a session tree from an export |
| GET | `/v1/sessions/{id}/tools` | `sessions:r` | Tool catalog for this session (409 if not loaded) |
| GET | `/v1/tasks` | `sessions:r` | Background tasks, newest first; `?session=<id>` keeps one session's, `?status=` one status |
| DELETE | `/v1/tasks/{task_id}` | `sessions:w` | Cancel a background task by its full id |
| GET | `/v1/jobs` | `schedule:r` | Scheduled jobs, every session's or one session's with `?session=<id>` |
| POST | `/v1/jobs` | `schedule:w` (+ `sessions:w` for a `gate`) | Create a scheduled job on the session `session_id` names |
| DELETE | `/v1/jobs/{job_id}` | `schedule:w` | Cancel a scheduled job by its full id |
| GET | `/v1/skills/{name}` | `skills:r` | One skill, with its body |
| PUT | `/v1/skills/{name}` | `skills:w` | Create or update a skill |
| DELETE | `/v1/skills/{name}` | `skills:w` | Delete a skill |
| GET | `/v1/memories` | `memory:r` | Memory index, or with `?q=<words>` the memories a search finds, best first; `referencing`, `unverified`, `expired` and `project` narrow either |
| GET | `/v1/memories/{name}` | `memory:r` | One memory, with its body; `?project=<name>` when the name is in several projects |
| PUT | `/v1/memories/{name}` | `memory:w` | Create or update a memory, in the tier `project` names |
| DELETE | `/v1/memories/{name}` | `memory:w` | Delete a memory; `?project=<name>` when the name is in several projects |
| GET | `/v1/mcp/{name}/tools` | `mcp:r` | Tools one MCP server advertises, each with its resolved permission, which step of the resolution chain decided it, whether config lets the agent see it, and whether a `readOnlyHint` the server sent was declined by `trust_read_only_hint = false` |
| POST | `/v1/mcp/{name}/reconnect` | `mcp:w` | Reconnect an MCP server |
| GET | `/v1/instructions` | `sessions:r` | Resolved standing instructions, without the per-session files `[instructions]` names; `writable` says whether a `PUT` lands, and `path` and `source` are shown to `config:r` or `config:w` |
| GET | `/v1/config` | `config:r` | Where the installation is (`config_path`, `data_dir`), when `config.toml` was last read whole and written, `config_changed`, and `restart_required` |
| POST | `/v1/config/reload` | `config:w` | Read `config.toml` again and serve what a write over HTTP would have: accounts, profiles, MCP servers brought to the file's entries, the standing instructions; name the sections only a restart applies; `422` when the file fails to parse or holds anything startup would refuse |
| PUT | `/v1/instructions` | `config:w` | Write the standing instructions to `instructions.md`, `{"content": "..."}`, blank to remove; `409` when another source is in charge |
| GET | `/v1/accounts` | `config:r` | Configured accounts, each with whether a credential is stored, and the credentials no account claims |
| GET | `/v1/accounts/{name}` | `config:r` | One account: every setting it states, and whether a credential is stored |
| POST | `/v1/accounts` | `config:w` | Add an account, with its `api_key` on an API-key backend; answers `201` with the account |
| PATCH | `/v1/accounts/{name}` | `config:w` | Rename an account, `{"name": "<new>"}`, moving its credential and every profile on it |
| DELETE | `/v1/accounts/{name}` | `config:w` | Remove an account and its credential; `422` while a profile names it |
| POST | `/v1/accounts/{name}/login` | `config:w` | Store a credential, `{"api_key": "<key>"}` on an API-key backend; with no body on a subscription backend, begin a browser login, `202` with the URL and a `login_id` |
| GET | `/v1/accounts/{name}/login/{login_id}` | `config:r` | Whether a browser login is `pending`, `completed` or `failed` |
| POST | `/v1/accounts/{name}/login/{login_id}` | `config:w` | Complete a browser login with what the browser gave, `{"code": "<code or callback URL>"}` |
| GET | `/v1/profiles` | read | Configured profiles |
| GET | `/v1/profiles/{name}` | `config:r` | One profile, with every setting it states |
| GET | `/v1/mcp/{name}` | `config:r` | One MCP server's entry as written, `required` and `deferral` as it runs under them with `inherited`, the kinds of credential stored, its connection state, and `error`, why the last connect failed, on a `failed` one |
| POST | `/v1/mcp` | `config:w` | Add an MCP server, the `[[mcp.servers]]` entry with `auth_token` or `client_secret` beside it; connects in the background; answers `201` |
| PATCH | `/v1/mcp/{name}` | `config:w` | Change an MCP server's keys in place (`null` removes one), restarting it, or applying a change to its tool policy alone on its connection; `disabled` stops or starts it |
| DELETE | `/v1/mcp/{name}` | `config:w` | Remove an MCP server, its connection, its tools from every session, and its credentials |
| POST | `/v1/mcp/{name}/login` | `config:w` | Store `{"auth_token"}` or `{"client_secret"}` for an MCP server; with no body, begin its OAuth login in the browser, `202` with the URL and a `login_id` |
| GET | `/v1/mcp/{name}/login/{login_id}` | `config:r` | Whether an MCP server's browser login is `pending`, `completed` or `failed` |
| POST | `/v1/mcp/{name}/login/{login_id}` | `config:w` | Complete an MCP server's browser login with the callback URL, `{"code": "<URL>"}` |
| POST | `/v1/mcp/{name}/logout` | `config:w` | Clear every credential stored for an MCP server |
| POST | `/v1/profiles` | `config:w` | Add a profile; `"default": true` makes it the default, and the second profile added names the one that was alone; answers `201` with the profile |
| PATCH | `/v1/profiles/{name}` | `config:w` | Change settings (`null` removes a key), make it the default, or rename it with `{"name": "<new>"}` sent alone |
| DELETE | `/v1/profiles/{name}` | `config:w` | Remove a profile; `422` while another's `fallbacks` names it, or while it is the default and two or more would remain |
| GET | `/v1/projects` | read | Every project, with when it was added and its memory and session counts |
| POST | `/v1/projects` | `memory:w` | Add a project, `{"name": "<name>"}`; answers `201` with the project |
| PATCH | `/v1/projects/{name}` | `memory:w` | Rename a project, `{"name": "<new>"}`, moving every session and memory on it |
| DELETE | `/v1/projects/{name}` | `memory:w` | Remove a project; `422` with the counts while a memory or session is on it |
| GET | `/v1/openapi.json` | None, and off unless `[serve].docs` is set | OpenAPI spec |

`GET /v1/sessions` is paginated by `limit` and `cursor` (see [Sessions](#sessions)), searches with `q` (see [Titles, pins and search](#titles-pins-and-search)), and takes `include_children=true` to list sub-agent sessions alongside root ones, `working_directory=<path>` to filter by working directory, and `parent=<id>` to list one session's direct children. A sub-agent's session record carries `parent_id`, which is what reconnects it to the session that dispatched it.

A memory record carries `project`, the [tier](memory.md#tiers) it is in (`""` for the global tier), `updated_at` (when the row last changed), `created_at` (when the memory was made, stamped once at creation) and `verified_at` (when the fact was last verified: the creation, until a write with `verified: true` moves it), plus its `load` (`inline`, `index` or `search`), its `references`, its `source`, its `expires_at` when an expiry was set, its `tags` and its `read_count`, how many times the agent has recalled it through `memory_read`. The timestamps are deliberately separate: a description edit moves `updated_at` without the note saying anything new, it is `created_at` that the model is shown as an age, and it is `verified_at` that search freshness decays from. `GET /v1/memories/{name}` adds `referenced_by`, the names of the memories whose references name this one.

`PUT /v1/memories/{name}` accepts `tags`, `references`, `source`, `load` and `expires` with the same omit-to-keep rule as `body`: omit to leave what the memory has alone, send `[]` or `""` to clear a list or a string. `project` is the tier to file the memory in, a project the store has or `""` for the global tier; omitted, an existing memory keeps its tier and a new one is global, and naming the other tier for an existing memory moves it. A name a global memory and a project memory would share, a project the store does not have, and a name that is in several projects when none is named are each a **422** carrying the store's sentence. `expires` is RFC 3339 or `YYYY-MM-DD` for the whole of that day. `verified: true` records that the write follows a check that the fact is current, and is the only thing that moves `verified_at` on an existing memory. A write that would take the inline bodies past [`[memory] inline_max_bytes`](../configuration/config-file.md#memory) answers **422** with the store's own sentence, naming the total, the largest inline memories and the remedy, and nothing of it lands; see [The inline budget](memory.md#the-inline-budget).

`GET /v1/memories` takes three filters beside `q`, the same three `memory_search` and `meka memory list` take: `referencing=<target>` for the memories whose references name a memory, path, URL or id (matched whole, without regard to case), `unverified=<duration>` for the memories not verified within that long (`90d`), and `expired=true` for those past their expiry. `project=<name>` keeps one tier (`project=` the global tier); without it every tier is read. Without `q` the filters list in index order; with it they narrow the search.

`GET /v1/memories?q=<words>` answers with the memories the agent's own `memory_search` would find for those words, best first and at most `limit` of them (default 10, clamped to 1..25), each carrying the `excerpt` it was found on; the same ladder of exact, prefix and substring matching, so an operator and the model find the same memories for the same words. Without `q` it is the whole index.

`GET /v1/memories/{name}` answers **404** for a name that is not stored, and **422** for a name that is in several projects when `?project=` names none; `DELETE` answers the same and removes nothing on a 422. Reading through this endpoint deliberately does *not* increment the memory's read count: an operator is not the agent recalling anything, and the count feeds search ranking.

Descriptions and bodies are returned **exactly as stored**, not as they are rendered into a model's context: this endpoint is a backup and inspection door, like `meka memory export`, and stripping characters out of a note on the way through would make a restore lossy. JSON escaping keeps that safe in transit, but a client that decodes and prints the text to a terminal should neutralize it, as meka does at its own render boundaries.

These four endpoints are **not** gated by `[memory] enabled`. That switch decides whether an agent keeps memories; a token is the operator, so it reaches a store that already exists exactly as `meka memory list` does in a shell.

A scheduled job's optional `gate` is the sharpest grant on this API, and how sharp depends on what it checks. It requires `sessions:w` in addition to `schedule:w` either way.

A **shell** gate (`"check": {"command": "…"}`) runs through `sh -c` as the user running `meka serve`, on a timer, *before* the turn and independently of it, so it needs no working provider and no model to execute. The session must be at `unrestricted`. This is the one grant `workspace` does not carry: the command runs outside the turn, so nothing confines it to the workspace roots, and the API's own 403 says `unrestricted`.

A **tool** gate (`"check": {"tool": "…", "arguments": {…}}`) is not held to that bar. It may only name a tool meka resolves to `read`, and the session need only be at `read`. Both facts are re-checked on every fire, so a tool that resolves higher after a config change stops being a gate.

`shell_execute` is one such tool wherever a sandbox backend is usable, so a `read` session can plant an arbitrary command on a timer through the tool form. That is deliberate and it is not the same grant as the shell form: a gate dispatches at `read`, the level meka sandboxes, so the command runs read-only-confined rather than as a bare `sh -c`, and where no sandbox is available the tool resolves above `read` and the gate is refused instead. The confinement blocks writes, not the network. See [Scheduled jobs](./scheduling.md) for the longer version.

No job of any kind can be created on a session at `none`, gated or not: no tool runs there, so the turn could neither act on the job nor cancel it, and `POST /v1/jobs` answers 403 `session-permission` rather than creating a row that can never run. A job whose session drops to `none` afterwards keeps its row and reports itself: every job view carries a `withheld` field, present only when something is holding the job back. With `sessions:r` it is the same sentence the agent is given; a `schedule:r`-only token gets a fixed sentence saying the reason needs `sessions:r`, because the reason can name the session's level, a gate's tool, or the first line of a check's output. It is computed per request from the session's current level, so it tracks a `PATCH /v1/sessions/{id}` without the job being rewritten.

A `schedule:*`-only token can still plant ordinary prompt-only jobs; it cannot reach a gate at all. Scope a bridge accordingly, and note that `GET /v1/jobs` is server-wide, so `schedule:r` alone lists every session id in the store.

`DELETE /v1/jobs/{job_id}` and `DELETE /v1/tasks/{task_id}` take the full id, which is what a client of this API holds; the 8-character short form `meka schedule cancel` and the `schedule_cancel` / `task_cancel` tools accept is a terminal's ergonomic and matches nothing here. An id matching nothing is a 404, so a typo is never reported as a cancellation. A job that a scheduler sweep retired a moment before the delete is a 404 as well, for the same reason: 204 means this request canceled the job, not merely that it is gone.

`GET /v1/tasks` lists every session's background tasks, newest first, `?session=<id>` keeps one session's (an empty list for a session with none, or an id no session has), and `?status=running` (or `completed`, `failed`, `canceled`, `interrupted`) keeps one status; a status that is none of those is a `422`. `DELETE /v1/tasks/{task_id}` answers `204` for a task it canceled or one already finished, and `404` when no task has the id; a sub-agent's task is canceled like any other. A task that is a backgrounded `agent_spawn` or `agent_followup` carries `subagent_id`, the session it runs, from the moment the call returns, so a dashboard connects the task to the sub-agent's record without parsing the task's label.

Canceling a background task records the cancellation and signals the running task, but only `meka serve` can signal work `meka serve` started. If the session is open in another process (a `meka -r` screen, say), the row is marked `canceled` and the command keeps running there until it ends on its own; its result is then discarded, because the row is no longer `running`.

A `connected` server in `GET /v1/mcp`'s `servers` carries `protocol_version`, the MCP revision its connection speaks: `2026-07-28` when the server answered meka's probe, otherwise the version the `initialize` handshake settled on (`2025-11-25` for a current server). The field is absent in every other state.

`POST /v1/mcp/{name}/reconnect` answers 200 with where the server now stands, which is not the same as "it worked": **read `state`, not the status code**. An attempt that ran and failed is a 200 carrying `state: "failed"`, not a 502. A server the startup sweep is still connecting comes back as `state: "pending"` with no attempt made, so a dashboard polling `GET /v1/mcp` during startup does not mistake "still coming up" for "down". The two non-200s are narrow: 422 when the server is `disabled` in config, and 502 when a connected server's new connection could not be opened, whether its connect failed or ran past `[mcp] connect_timeout`. A connected server is given a new connection, which runs its `headers_helper` again, and keeps the one it has when the new one cannot be opened. Neither carries the reason, which is in the meka log: a connector's reason has quoted a spawn command line and a URL with its key.

An MCP server's OAuth login runs over HTTP as an account's does, under `config:w`; see [Managing the installation](#managing-the-installation). A token holding `mcp:w` alone reconnects a server and nothing more.

For full request/response schemas, see `/v1/openapi.json` on a running server, or browse it in one of the [tools below](#import-into-tools).

It is **off unless you set [`[serve].docs`](../configuration/config-file.md#servedocs)**, and unauthenticated when on, so CI pipelines and code generators can fetch the spec without a token. That combination is what makes it opt-in: it takes no token *and* it publishes the shape of every endpoint the deployment exposes, which is useful on a workstation and reconnaissance anywhere else.

### Exporting the spec

Save a local copy for offline use or code generation:

```bash
curl -s http://localhost:8080/v1/openapi.json -o openapi.json
```

### Code generation

Generate a typed client from the exported spec:

```bash
# Python (openapi-python-client)
openapi-python-client generate --path openapi.json

# TypeScript (openapi-typescript)
npx openapi-typescript openapi.json -o src/api.d.ts

# Go (oapi-codegen)
oapi-codegen -package api openapi.json > api/api.gen.go

# Rust (progenitor)
cargo progenitor-client openapi.json
```

### Import into tools

- **Postman / Insomnia:** Import → URL → `http://localhost:8080/v1/openapi.json`
- **Bruno:** Create collection from OpenAPI → paste the URL or a saved file.
- **Swagger Editor:** File → Import URL → `http://localhost:8080/v1/openapi.json`
