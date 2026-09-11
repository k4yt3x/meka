# Upgrading

Most upgrades are a binary swap: replace the old executable with the new one and carry on. This page covers the ones that are not.

## Copying a store

The store under `MEKA_DATA_DIR` runs in WAL mode, so the most recent writes, including a schema
migration, can sit in `meka.db-wal` beside `meka.db` until SQLite checkpoints them. A copy that
takes `meka.db` alone can therefore carry a schema version its tables have not caught up with.
meka checks for that on open and refuses the store rather than running against it. Copy the `-wal`
and `-shm` companions with the file.

## 0.49 to 0.50

**`meka tools` is `meka tool`.** Every top-level command names the object it manages in the
singular, and this was the one that did not. `meka tools list` is `meka tool list`; the flags and
the JSON envelope are unchanged. `meka mcp tools <name>` keeps its name, since it lists the tools
of a server rather than managing tools.

**`/tasks` is `/task` in the REPL**, for the same reason: `/skill`, `/memory` and `/schedule` are
singular. `/task`, `/task show <id>`, `/task cancel <id>` and `/task cancel --all` do what the
plural did. The HTTP route `/v1/sessions/{id}/tasks` is unchanged.

## 0.46 to 0.47

**`search_web` is gone.** It scraped DuckDuckGo's HTML and was turned away by the bot detection more
often than not. Web search now comes from an MCP server, which packages one engine's key, quota and
response shape without meka embedding any of them:

```bash
meka mcp add exa https://mcp.exa.ai/mcp
```

A `[tools]` list that still names `search_web` warns at startup and otherwise works. Existing sessions
that called it resume unchanged: their tool results are text, and nothing re-runs them.

## 0.45 to 0.46

The store migrates itself, as every release since 0.43 has. **`config.toml` does not**, and this
release changes its shape: a `[providers.<name>]` profile is now an `[accounts.<name>]` table plus
a `[profiles.<name>]` table, `default_provider` is `default_profile`, and the `ask` permission
level is gone. A config in the old shape is refused at startup, naming the first key meka does not
know, rather than read with a guess at what it meant. The conversion is a one-shot script,
`migrate-0.45-to-0.46.py`, attached as an asset to the 0.46 release.

Beyond the config shape, this release renames several tool parameters, changes a handful of HTTP
fields and status codes, and makes ACP answer `InvalidParams` where it answered `InternalError`.
Everything a client, a skill or a script could depend on is listed below under "What else changed"
and "Tool parameters", each with its remedy. Run the script, launch once, and work down those two
lists for anything you automated.

### Order

1. **Run the script against your config, first as a dry run, then with `--apply`.** It needs
   Python 3.11 and the `tomlkit` package (`pip install tomlkit`), which is what lets it keep every
   comment and the order of everything it does not touch.

   ```bash
   python3 migrate-0.45-to-0.46.py            # prints a diff; writes nothing
   python3 migrate-0.45-to-0.46.py --apply    # rewrites config.toml in place
   ```

   It finds meka's config the way meka does, honoring `MEKA_CONFIG_DIR`; `--config PATH` points
   it at a copy instead. `--self-test` checks the script against its own fixture and exits.
2. **Install 0.46 and launch it once.** The store migrates on that open, behind an automatic copy
   beside it named for the schema version it came from (`meka.db.v9.bak` for a store 0.45 left),
   in ten ledger steps. The first creates the REPL's `prompt_history` table where a store lacks
   one, a no-op otherwise. The other nine: `sessions.provider` becomes `sessions.profile`, and
   `provider_credentials` becomes `account_credentials`, keyed by account, both renames of what
   was always there; an `approvals` column is added, every `ask` session becomes `none` with it
   on, and a root session that never recorded a level (one an ACP client created) adopts
   `[permissions].default`; each stored turn's inline `<context>` preamble becomes its own
   `turn_context` block; every image's bytes move out of its message row into `blobs`, leaving
   a reference; every column and index takes one naming rule, with the JSON in two of them
   following suit; a `repair` row's own thinking blocks take that rule's tag too, which the
   step before it passed over; a root session still without a level after all that takes
   `[permissions].default`, and a launch that cannot read the file refuses here rather than
   skipping the stamp; and a stopped task's stored status is spelled `canceled`. Four of these
   walk every message row, so a store with years of
   image-heavy sessions takes a moment on that first launch and grows a copy of the same size
   beside it. A store restored from a `.dump` replays the whole ledger, and if no default profile
   can be resolved when it does, the frozen 0.44 step warns with its old `--provider` advice; read
   it as `--profile`.

Run the script before you launch 0.46, not after. A `meka` launched against an unconverted config
warns that it cannot read the file and then refuses whatever needed it; only the commands that
edit it through `toml_edit` (`meka account remove`, `meka profile remove`, `meka mcp remove`) and
the ones that read the store alone still run. The store migrates on that launch only if no step
needs the file. A root session that never recorded a level needs `[permissions].default` from it,
and a migration that cannot read the file refuses and rolls back rather than stamping nothing, so
the store keeps its 0.45 shape until the launch after the script has run.

### What the script converts

| Before | After |
|---|---|
| `default_provider = "work"` | `default_profile = "work"` |
| `[providers.work]` with `type`, `base_url`, `client_id`, `oauth_token_url`, `device_id` | `[accounts.work]` with `backend` in place of `type`, and the other four unchanged |
| `[providers.work]` with `model`, `context_window`, `max_output_tokens`, `effort`, `vision`, `thinking`, `thinking_budget`, `max_request_bytes`, `redact_thinking` | `[profiles.work]` with `account = "work"` and eight keys unchanged; `redact_thinking = true` becomes `thinking_display = "redacted"` and `false` becomes `"summarized"` |
| `[permissions].enabled` containing `"ask"` | `"none"` in its place |
| `[permissions].default = "ask"` | `default = "none"` and `approvals = true` |
| `[web].request_timeout_seconds = 30`, `connect_timeout_seconds`, `read_timeout_seconds` | `request_timeout = "30s"`, `connect_timeout`, `read_timeout`, by value; a `0`, which meant the default, is removed |
| `[mcp].grace_seconds = 3`, `connect_timeout_seconds = 30` | `grace = "3s"`, `connect_timeout = "30s"`, by value; a `0` becomes `"0s"`, which `grace` accepts and `connect_timeout` refuses at startup |
| `[mcp].strict` | `default_required`, same meaning |
| `[session].retention_days = 30` | `retention = "30d"`, by value |
| `[thinking].budget_tokens` | `budget`, same value |

Every profile becomes one account and one profile of the same name, so nothing you named changes
its name and every session still resolves. Two old profiles on one login stay two accounts with two
copies of the credential; merge them by hand if you like, by pointing both profiles' `account` at
one and running `meka account remove` on the other once nothing names it. A key the script does not
know is carried into the profile table and reported, where meka will refuse it by name; a duration
key whose value is not a whole number is left under its old name and reported, with the same result.

### What else changed

- **The `meka provider` suite is gone.** `meka account add`/`login`/`list`/`remove` manage
  accounts and their credentials; `meka profile add`/`set`/`use`/`list`/`remove` manage profiles.
  `meka account usage`/`whoami`/`stats` are where they were, and take `--profile <name>` instead of
  a positional name. `account add` takes `--backend` where `provider add` took `--type`.
- **The prompt is a flag.** `meka "text"` is `meka -p "text"`, and `-p -` reads the prompt from
  stdin. There is no positional prompt, so `meka unknowncommand` is an error rather than a session.
- **`--provider` is `--profile`**, long form only: `-p` is the prompt.
- **`--format json` on a `--oneshot` run** prints one object for the turn; see [One-shot
  mode](../usage/one-shot-mode.md#json-output).
- **HTTP API**: the `provider` field on `POST /v1/sessions`, `PATCH /v1/sessions/{id}` and every
  session response is `profile`, and `GET /v1/providers` is `GET /v1/profiles`, whose rows carry
  `account` and `backend` in place of `type`. A session export archive's `provider` field is
  `profile`, and its `format_version` is 2, so an archive written by 0.45 is refused by version;
  re-export it from a migrated store.
- **ACP**: the `configOptions` entry `provider` is `profile`.
- **REPL**: `/provider` is `/profile`, and `/status` shows the profile with its account.
- **One spelling per value.** `--permission` and `MEKA_PERMISSION` take a level's full name (`n`,
  `r`, `w`, `u` are gone), and `--render-mode`, `MEKA_RENDER_MODE` and `[display].render_mode` take
  `termimad`, `syntect` or `raw` (`rich` is gone). The flags and the config key refuse anything
  else; the two variables warn and fall through to the next source, as they always have. The
  undocumented `text` spelling of `--format plain` is gone too.
- **A `[permissions].default` or `enabled` entry naming a level meka does not have is refused at
  startup**, with the line, the way an unknown key is, instead of being dropped with a warning. The
  script rewrites `ask`; anything else you spelled yourself.
- **The `ask` permission level is gone**, replaced by the `approvals` switch beside the level: a
  call above the level is refused, or put to you when the switch is on. `/approvals on|off` in the
  REPL, `approvals` on `POST /v1/sessions` and `PATCH /v1/sessions/{id}`, and the ACP config option
  of the same name set it; `[permissions].approvals` is what a new session starts with. The store
  migration turns an `ask` session into `none` with approvals on, which asks about every call as
  `ask` did. An approved call now runs *at the session's level*, so an approved write at `read`
  lands only under the workspace roots where `ask` wrote anywhere; raise the level if an approved
  call needs the reach. See [Permissions](../usage/permissions.md#approvals).
- **A user message is two blocks.** What meka injects ahead of the words for a turn (permission
  and environment context, todos, catalog changes, background outcomes, the resume notice) is its
  own `turn_context` content block, first, and the words are a `text` block. `GET
  /v1/sessions/{id}/messages` returns the block typed, so a client reading `content[0].text` as the
  prompt now reads the context; take the `text` blocks. A migration splits every stored turn once.
- **Image bytes live in a `blobs` table.** The migration moves every inline image out of its
  message row and leaves a reference by content hash, so a screenshot read twice is stored once. A
  session export carries a `blobs` list with the bytes its sessions reference, and an archive that
  references a blob neither it nor the store holds is refused. Over HTTP an image block reports
  `media_type` and `hash`, and `GET /v1/sessions/{id}/blobs/{hash}` serves the bytes.
- **A scheduled job runs at its session's recorded level and nothing else.** The polling process's
  own `--permission` no longer stands in for a session row that records no level; every surface
  records one at creation, and the migration stamps the configured default on any older root row
  that never got one. A sub-agent's row now records its level too.
- **Every config duration is a humantime string.** `[web].request_timeout`, `connect_timeout` and
  `read_timeout`, `[mcp].grace` and `connect_timeout`, `[session].retention` (`"30d"`); the script
  converts the `_seconds` and `_days` keys by value, and `"0s"` is refused where zero is
  meaningless. `[mcp].strict` is `default_required` and `[thinking].budget_tokens` is `budget`,
  both converted.
  `MEKA_MCP_STDIO_CONCURRENCY` and `MEKA_MCP_HTTP_CONCURRENCY` are gone: set
  `[mcp].stdio_concurrency` and `http_concurrency` (3 and 20 by default, zero refused).
  `MEKA_MCP_TOOL_TIMEOUT` takes a duration such as `10m`, not milliseconds; a bare number is
  ignored with a warning and the default of ten minutes applies.
- **One exact spelling per value, everywhere.** `--permission`, `--render-mode`,
  `--sandbox-backend`, `--format` and `mcp add --transport` refuse case variants (`Read`, `JSON`),
  `session export` and `GET /v1/sessions/{id}/export` drop the `md` alias of `markdown`, `mcp add
  --auth` takes `oauth`, `client_credentials` or `client_credentials_jwt` as the `[auth]` block
  spells them (the hyphenated forms are gone), and `[mcp].default_permission`, a server's
  `permission` and `tool_permissions`, and `[tools].tool_permissions` refuse a level meka does not
  have at startup, naming the line, where they warned and ignored it.
- **HTTP API**: an unloaded session whose row records no level omits `permission` (it sent `""`);
  every optional field is omitted rather than `null`, `display_summary` included; `GET
  /v1/health/ready` reports `profile_configured` (was `provider_configured`); the
  `permission_required` event carries `input` and stays answerable for 30 minutes (was 60 seconds);
  a body that fails to parse says which field on every endpoint; `POST
  /v1/sessions/{id}/responses/{request_id}` at a sub-agent's id answers 422 `session-not-drivable`;
  a fork of a session another meka process holds answers 409 `session-locked`; a session's `title`
  is the first user words with whitespace collapsed, cut at 80 characters, and `meka session show`
  labels it `title` (was `opening`). Four status codes move: a `[web]` or `base_url`
  misconfiguration is a sanitized 500 (was a 422 naming the operator's path), and a session lock
  meka cannot open is 500 (was 409 `session-locked`); meka's own request-ceiling refusal is 422
  with the new `type` `request-too-large` (was 502 `provider`); `GET` and `PATCH
  /v1/sessions/{id}` answer 404 or 500 for a row they cannot read (was 200 with `profile: ""`);
  and `POST /v1/sessions/{id}/schedule` with scheduling disabled is 404 (was 422).
- **ACP**: a locked session, a sub-agent's id, a profile the config no longer has and every other
  refusal the caller can act on answer `InvalidParams` (was `InternalError`);
  `session/set_config_option` refuses a profile switch while a turn is in flight instead of writing
  the row and deferring; `session/new`, `load`, `resume` and `fork` refuse a `cwd` that is not an
  existing directory and record it canonically; tool-call and permission titles read `<tool_name>
  <argument>` (`read_file src/x`) and permission requests carry `rawInput` with a JSON content block.
- **Terminal output**: every timestamp is local time with its UTC offset (`2026-09-07 14:03
  +02:00`), sizes print as MiB, KiB or B, and every listing command takes `--format json`, printing
  the HTTP API's record shapes. Tool-call indicators and the approval prompt show a tool's real
  name (`read_file`, not `ReadFile`); the prompt is headed `[approval]` and takes `always` and
  `never`.
- **Skills you wrote** that name a renamed tool parameter (next table) or the old `[ask]` prompt
  must be edited by hand; meka does not rewrite skill files.
- **A gate's pointer test is `not_empty`** (was `not-empty`) in `schedule_create`, `POST
  /v1/sessions/{id}/schedule` and `meka schedule add`; stored jobs are converted by the store.
- **`canceled`, one `l`, on every wire meka owns.** Match `turn.canceled` as the SSE terminal event,
  `https://meka.so/errors/turn-canceled` as the problem `type`, and `status == "canceled"` in task
  views (`GET /v1/sessions/{id}/tasks`, `DELETE .../tasks/{task_id}`), `task_list` output and
  `schedule.fired` webhook bodies; the `reason` values are unchanged. The store rewrites its stored
  task rows on first open (the tenth ledger step). ACP's `stopReason: "cancelled"` and MCP's
  `notifications/cancelled` are those protocols' own spellings and stay.

### Tool parameters

Six built-in tool parameters are renamed so that one name means one thing across the catalog:
`is_regex` for a boolean, `glob` for a glob, `limit` for a result cap, `id` for an identifier.

| Tool | Before | After |
|---|---|---|
| `conversation_search` | `regex` (boolean) | `is_regex` |
| `conversation_read` | `count` | `limit` |
| `find_files` | `pattern` | `glob` |
| `fetch_url` | `max_length` | `limit` |
| `agent_followup` | `agent` | `id` |
| `agent_delete` | `agent` | `id` |

A call spelling the old name is missing its required parameter (`glob`, `id`) or, where the
parameter was optional, has it ignored in favor of the default. `search_contents` gains a `limit`
(1 to 100, default 100) beside its unchanged `pattern`.

meka does not rewrite what names these. A skill under the skills directory
(`~/.config/meka/skills/<name>/SKILL.md`) that spells out a `find_files` or `agent_followup` call
must be edited by hand, and a scheduled job whose gate calls one of these tools with the old
argument must be recreated. Past calls in a session's history keep the old names, which is
harmless: the model reads the current schema on its next turn.

The sections below predate 0.46 and use its old names: `--provider` is `--profile`, `meka provider
…` is `meka account …` and `meka profile …`, the positional prompt is `-p`, and the `ask` level is
`none` with approvals on.

## 0.43 to 0.44

A binary swap, and the store migrates itself as promised below, **unless you authenticate an MCP
server with `auth_token` or `client_secret`**, which are no longer config keys. Read the next
section first if you do; meka will refuse to start otherwise. Then the behavior changes below,
worth reading before you resume an existing session or run a scripted `meka`, several of which apply
only if you run `meka serve` or `meka acp`.

**MCP secrets moved out of `config.toml`.** `auth_token` on a server, and `client_secret` in a
`[mcp.servers.auth]` block, are gone. Both were secrets sitting in a plaintext file people commit
and sync; they now live in the store beside the OAuth tokens, which is where the login
credentials have always been.

meka cannot move them for you. The store migrates itself because it has a ledger recording what it
has already done; `config.toml` has none and may be older or newer than the binary at any moment, so
a key left behind is a parse error naming the key and the line rather than a value silently ignored:

```console
$ meka mcp list
Error: database error: schema migration 3 ('sessions_name_their_provider') failed: Invalid
parameter name: cannot record a provider for 4 carried-forward session(s) while config.toml
cannot be read; fix the file and start meka again. The store is unchanged
```

The parse error itself is a warning just above it, naming the key and the line:

```console
WARN meka: failed to read config.toml, so no profile can be adopted for older sessions:
configuration error: failed to parse …/config.toml: TOML parse error at line 12, column 1
   |
12 | auth_token = "…"
   | ^^^^^^^^^^
unknown field `auth_token`, expected one of `name`, `transport`, …
```

Two messages because two things are stuck: the file will not parse, and the migration that has to
name a profile for your existing sessions cannot ask it which one. Fixing the file fixes both, and
nothing has been written in the meantime: "The store is unchanged" is literal, and the copy taken
before the attempt is still beside your store. (On an installation with no sessions to carry
forward, only the parse error appears.)

For each server, delete the line and store the secret instead. Which command depends on which key
you deleted, and the two are alternatives, not a sequence: a bearer belongs to a server with no
`[auth]` block, a client secret to one that has it.

```console
$ # for a server whose `auth_token` you deleted (no [auth] block):
$ pass show api-token | meka mcp login api --auth-token-stdin

$ # for a server whose [auth] block's `client_secret` you deleted:
$ pass show acme-secret | meka mcp login acme --client-secret-stdin
```

`meka mcp get <name>` then lists the kinds it holds without printing any of them. `--auth-token` and
`--client-secret` are gone from `meka mcp add` for the same reason: an argument is visible in `ps`
output and in the shell history of every user on the machine. Use the `-stdin` forms, which `add`
also takes.

If you were using `auth_token = "${API_TOKEN}"` to keep the token out of the file, a header does the
same job and still expands: `headers = { Authorization = "Bearer ${API_TOKEN}" }`. Storing it is the
better answer, since it survives without the variable being set.

Nothing else about a server moves. `env`, `args` and `headers` stay in `config.toml` with `${VAR}`
expansion, because they configure a process or a request and merely *may* contain a secret.

**`isolated` scheduled jobs are gone; every job fires in the session that created it.** The mode ran
a job's turn in a fresh session rather than the conversation that made it, to avoid replaying that
conversation's history. Only `meka serve` ever honored it: the REPL and ACP already ran such a job
in the open conversation, with a warning, so for two of the three hosts nothing changes at all.

Existing jobs are not deleted and do not need touching. The store drops the column and the job keeps
its schedule and its prompt, firing into the session it belongs to from then on.

What it cost is why it went. The fire inherited the creating session's authority (its permission
level, its working directory, its profile, its MCP servers) and dropped the conversation,
which is where anything you told the agent that never reached a memory or an instructions file
lives. Its result landed in a session nothing linked to, and the turn could not even cancel its own
job, because `schedule_cancel` resolves against the session it is running in.

`meka acp` and `meka serve` clients: `POST /v1/sessions/{id}/schedule` now refuses `isolated` with a
422 naming the field, rather than accepting and ignoring it. `GET /v1/schedule` and the
`schedule.fired` webhook no longer carry it either.

If you were relying on the mode, an external timer does the same job with the level and profile
stated outright instead of inherited (0.44 syntax):

```bash
meka --oneshot --permission read --provider work "summarize today's alerts"
```

Often a gate is the better answer: it means a frequent job takes no turn at all on the ticks where
nothing happened, which saves more than skipping the history did.

**A session another one spawned is driven only by its parent.** `POST /v1/sessions/{id}/turn`
answers 422 for a sub-agent's id, `meka -r <sub-agent-id>` refuses by name, and a scheduled fire aimed
at one does the same. Both agent builders now check, rather than the scheduling door alone.

What this closes is that a sub-agent's restrictions live in its spawn record, which those builders
never read: the `[subagents]` denials it was created under, its memory and instruction grants, and
the permission ceiling its spawn call set. Driving one from a host therefore ran a conversation that
was *deliberately given narrow tools* with the full built-in set at the host's level. `agent_followup`
was and remains the door that reconstructs those terms, so nothing meka does for you changes.

Reading a sub-agent is untouched: `meka session export`, `GET /v1/sessions/{id}/messages` and
`meka session list --include-children` all still serve it.

**Forking one does not promote it**, and that is the other half of the change. A fork of a sub-agent
now carries `parent_session_id` and the spawn terms, so the copy is a sibling under the same parent
rather than a new root; without that, `POST /v1/sessions/{id}/fork` was a one-call way around the
refusal above, handing back a live session over a sub-agent's whole conversation with none of the terms
it was spawned under. The two doors that have to hand back a *live* session therefore refuse a
sub-agent's id up front: `POST /v1/sessions/{id}/fork` answers 422, and ACP's `session/fork` answers
`InvalidParams`. `meka session fork` still makes the copy: it takes no runtime, and the copy is
readable like any other sub-agent. Forking an ordinary session is unchanged. If you want a sub-agent's conversation as a root session of your own, copy
the text out rather than expecting a command to promote it.

**`meka session list --long` is gone**, along with the columns it showed. If a script parses that
output, it needs updating; the default columns are unchanged.

**`/cd` with no argument returns to the directory meka was launched from**, not `$HOME`. `/cd ~`
still goes home. The old behavior made a bare `/cd` a surprising way to leave the project you were
working in.

**`render_mode = "silent"` is gone**, as are `--render-mode silent` and `MEKA_RENDER_MODE=silent`.
Delete the setting: `termimad` is the default.

A config still carrying it fails to parse, naming the value and the line, and `--render-mode silent`
is refused by clap. `MEKA_RENDER_MODE=silent` is the quiet one: an unreadable value there has always
been dropped in favor of the next source, so it falls through to your config file or the default
rather than saying anything.

It never did what it says. It suppressed the model's answer and nothing else, so a run under it
printed the session id, the reasoning line, tool indicators, todo lists, notices and token usage,
and dropped the one thing you were waiting for. Both things it might plausibly have meant are
shell redirections that already work, and work the right way round: `meka … 2>/dev/null` keeps the
answer and drops the chrome, `meka … >/dev/null 2>&1` drops both.

**SSE `thinking.delta` now carries one chunk of reasoning per event.** It used to send one event per
completed block, so a client that opted into `supports_reasoning_stream` and rendered each event as a
whole block will now show fragments. Concatenate the deltas to rebuild the block, exactly as you
already do for `assistant_text.delta`. A client that concatenated needs no change, and one that never
set the capability sees nothing new. A turn the provider answered without streaming still arrives as
a single delta, so there are no two shapes to tell apart, and `stream: false` still reports each block
whole in `thinking`.

One consequence worth planning for: a session receiving reasoning gives up its retry on a transient
provider failure, because the deltas have already reached you and a second attempt would repeat them.
Leave `supports_reasoning_stream` off if you would rather have the retry.

**`meka session delete` refuses ids given alongside `--all`.** It used to take both and quietly do
the wider thing, so `meka session delete "$ID" --all` with `$ID` unset deleted every session and
then reported the empty id as a failure: a complete wipe reported as an error. Naming sessions and
asking for all of them are two different requests; say one or the other. `--older-than-days` has
conflicted with both for the same reason since 0.44.

**Every command taking a session, job or task id now accepts a unique prefix of one**, which is what
the listings print. Full ids still work, so nothing that already worked stops. An ambiguous prefix
is refused with the candidates named, and an empty one matches nothing rather than the only row:
`meka schedule cancel "$JOB"` with `$JOB` unset used to cancel whatever job was alone.

**`meka mcp logout <name>` clears every credential that server holds**, not only its OAuth tokens.
If you were using it to drop a stale token from a server that also has a stored bearer or client
secret, you will now need to store that again with `meka mcp login`.

**A scheduled job is refused on a sub-agent session.** `POST /v1/sessions/{id}/schedule` answers 422
if the session was spawned by another. Sub-agents never had the `schedule_*` tools, so no job meka
created can be affected; what this closes is a client planting one directly, which would have woken
the sub-agent without the tool restrictions or memory grants it was spawned under.

**ACP `session/load` and `session/resume` refuse a sub-agent's id.** Both used to take the session's
lock, rewrite its `cwd`, retire its background work and replace its roots before failing with
`Internal error`; both now decline with `InvalidParams` before touching anything, naming the parent
to use `agent_followup` from. An editor that stored a sub-agent's id from `session/list` gets a clear
refusal instead of a mutated row and an opaque failure. Over HTTP the same holds for every write-side
endpoint: `POST /v1/sessions/{id}/turn` and its neighbors refuse before taking the sub-agent's lock or
marking its background tasks interrupted.

**A session that carries spawn terms is refused even when its parent is not in the store.** That
shape has one source, and it is a pair of documented commands: `meka session export <sub-agent>
--format json` followed by `meka session import`. The archive's `parent_id` points outside it, so the
import re-roots the row while copying the spawn terms faithfully. The result reads as a sub-agent's
conversation to every door, so `meka -r` on it, and `POST /turn`, `/fork`, `/schedule` and `PATCH`
against it over HTTP, all refuse. `meka -c` skips it and `meka session list` shows it only under
`--include-children`, both so nothing offers you a session it will then decline. **If you were using
export-then-import to promote a sub-agent into a standalone session, that no longer works**; there is
no supported replacement, because the tools and permission ceiling a sub-agent ran under live in the
terms its parent set and nothing outside that parent can reconstruct them. The conversation itself
stays fully readable, and importing a *whole tree* (the root and its sub-agents together) is
unaffected, since each child keeps its parent.

**This upgrade deletes the pre-migration copy 0.43 left, and keeps one from now on.** Before it
migrates, meka copies the store aside; until now nothing removed those, so a full duplicate of your
whole history accumulated per schema-changing release. From 0.44 a fresh copy supersedes the one
before it. In practice that means `meka.db.v1.bak` in your data directory, the copy of your
pre-0.43 store, is removed on this upgrade and replaced by a copy of your pre-0.44 one. **If you
want the older file, move it somewhere else before upgrading.**

Two things worth knowing about what is kept. Peak disk during an upgrade is higher than the steady
state, because the new copy is written before the old ones go: budget for the store plus every copy
already beside it plus one more, and expect to settle back at twice the store. And the copy is taken
per schema-*changing* upgrade, not per release, so one file can span several versions if you skip
some.

**What keeping only the newest copy costs, stated plainly.** The copy you hold is of the store
*after* the migration before this one. So it undoes the most recent conversion and nothing earlier:
if a migration converts something wrongly, you do not notice, and you then take another
schema-changing upgrade, the only copy predating the fault is gone. That is a real limitation rather
than a technicality, and it is the reason to move a copy of your own aside if a particular upgrade
worries you. It is accepted because the alternative was an unbounded pile of full-size duplicates,
whose cost is certain where this one is conditional on a bug outliving a release.

**A resume now starts at the level the session recorded.** Both CLI hosts do this: the REPL and
`meka --oneshot -c` / `-r`. The scripted one is where a silent change matters most, since a
`--oneshot` run that passes no `--permission` used to start at the config default and now starts at
whatever the session was last set to. A session you created with `--permission unrestricted` comes
back at `unrestricted` without the flag. Before, the row said one thing and the run did another;
every other surface already read the row, and these two were the ones that did not. Pass
`--permission` on the resume to move it. A level that is no longer in `[permissions].enabled` is not
granted: the session drops to the configured default with a warning.

**A session now runs on the profile it was created with.** Every existing session is
recorded as running on your current default profile, which is what they were in fact running on, so
nothing moves. From here `meka -p openai` then `meka -c` stays on `openai`. If nothing could be
resolved when the migration ran (no profile configured yet), sessions are left without one and say
so; resume such a session once with `--provider <name>` to record it. The migration says which
profile it recorded and on how many sessions; run once with `-v` if you want to see it.

**A `502` from `meka serve` now carries the provider's own response text**, as a `provider_response`
member on the Problem Detail. It used to be withheld and written only to the server log.

The reason for the change is that the redaction defended less than it appeared to: `meka acp` has
always handed the same text to its client, so withholding it on HTTP left the text just as public
while making the one surface quieter. What it cost was the upstream's error type, which is the one
part of a failed turn a client can act on.

**Know who can read it before you leave it on.** An upstream refusal can name your account with the provider,
your organization, and your rate-limit posture. Submitting a turn takes `sessions:w`, but the
failure is also carried by the terminal `turn.failed` event, and re-attaching to a stream takes only
`sessions:r`, so a read-only token sees it too. If you issue read-only tokens to people who may
observe a session but are not entitled to the account behind it, set `[serve]
relay_provider_errors = false`. Nothing else changes: `detail` carries the same sentence either way,
and with the key off the member is simply absent.

The `503` for a required MCP server that is down is not affected and still reports only the server
names. That reason is meka's own subprocess text and has carried a command line and its filesystem
path, which is a different disclosure and not one this key governs.

**`GET /v1/info` no longer returns `provider` or `model`.** Read them from `GET /v1/providers`
instead, which lists every configured profile with its `name`, its `type` (the backend), its
`model`, and `active: true` on the one a session gets when it names none. The old fields held the
default profile's *backend* under the name `provider`, while `provider` on `POST /v1/sessions` names
a *profile*, so a client that read one and posted it to the other got a 422. They were duplicates of
the `active` row besides.

**If you ran a 0.44 development build, your store repairs itself on the next run.** One such build
removed a migration from the middle of the ledger instead of appending its reversal. `user_version`
is a positional index, so that renumbered every later step, and a store sitting between the hole and
the new head skipped a step it had never run while stamping itself current. The symptom was every
MCP connection failing with `no such table: mcp_credentials` after a migration that reported
success.

Nothing is needed from you: an appended step recreates the table and carries the old MCP credentials
into it, because a store already stamped past the missed step is only reachable by appending.
Released 0.43 stores were never affected; they sit at the baseline and migrate straight through.

**`--model`, `--base-url`, `--thinking` and `--thinking-budget` are gone.** A profile is an
indivisible bundle: the backend, the endpoint, the credential keyed to it, the model, and every
model-tied setting. A session selects one by name and records that name. A flag that moved one field
of the bundle left the rest behind, so `--model` against a profile stating `context_window = 1000000`
ran a 200K model while gauging its context against a 1M window, and never auto-compacted.

Change a setting on the profile (0.44 syntax):

```bash
meka provider set work model claude-opus-5
meka provider set work effort --unset
```

Or make a second profile and select it with `--provider`, which is now the only provider flag on a
run. `meka provider add` has a flag for every profile field except `device_id`, which meka resolves
and persists itself, so one command creates a whole profile (0.44 syntax):

```bash
printf '%s' "$ANTHROPIC_API_KEY" | meka provider add fast \
    --type anthropic-messages --model claude-haiku-4-5 \
    --context-window 200000 --api-key-stdin
```

**The thinking budget is per profile.** `[providers.<name>].thinking_budget` takes precedence over
`[thinking].budget_tokens`, which stays as the installation-wide fallback and needs no edit. The
global was previously cross-checked against a *per-profile* `max_output_tokens`, so a profile could
be refused over a number stated nowhere in it, and told to fix it by lowering a value every other
profile also read.

**`meka acp` and `meka serve` refuse `-c` and `-r`.** Both name one run's session, and a long-lived
host has no such thing: it creates one per `session/new` or per `POST /v1/sessions`, each naming its
own profile. They used to be accepted and quietly misapplied: `-c` / `-r` switched off the
default-profile check a host with no default needs most. Over HTTP, name a `provider` on `POST
/v1/sessions`; under ACP, `session/new` creates on the host's default and `session/set_config_option`
moves it.
`--provider` is still accepted, because it selects which configured profile the host defaults to,
which is a property of the host rather than of one session.

Also on the HTTP side, and not a break: `PATCH /v1/sessions/{id}` with a body naming only a provider
now works on a session that is not loaded, which is how you move one whose profile has left
`config.toml`. It takes the session lock to do it, so if you run more than one `meka` on the same
store, send it to whichever process has the session; another one answers `409` `session-locked`
rather than moving a row the running host would ignore.

## 0.42 to 0.43

Nothing to do. Start 0.43 and it brings the store forward itself, on the first open, before anything reads it.

This is the first release that migrates its own store, and from here on that is the rule: upgrades from 0.43 onward are a binary swap, whatever the schema does.

What it changes, if you want to know what happened. A scheduled job's gate used to be two columns, `gate_command` and `gate_fire`; it is now `gate_kind` plus a JSON `gate_spec`, which is what lets a gate call a read-only tool instead of a shell command. And a due job is now claimed by *leasing* it rather than by consuming its row, which adds `claimed_by`, `claimed_until` and `attempts`, so a host that crashes mid-delivery no longer loses the occurrence, or for a one-shot the whole job. Each gate's stored baseline is preserved, so a `changed` gate does not fire spuriously on its first evaluation afterwards.

Before it writes anything, meka copies the store to `meka.db.v1.bak` beside it. That doubles the space the store takes until you delete it, which is worth knowing if yours is large. Start with `-v` once if you want the exact path in the log; the copy is otherwise silent. It records the version it was taken at, so if you ever restore it, the next start migrates it again correctly rather than mistaking it for a store that is already current.

The whole thing is one transaction, so an interruption leaves the store exactly as it was rather than half-converted. Running two hosts at once is fine: the first takes the schema lock and the second waits, then finds nothing to do.

**Coming from 0.41 or older**, run `migrate-0.41-to-0.42.py` once first, as described below. 0.43 recognizes a 0.41-shaped store and refuses it by name rather than converting it into something still unreadable, and it changes nothing when it does.

### A gate that cannot be read

Rare, and worth knowing the shape of. If a job's gate was already unreadable under 0.42 (a hand-edited row, or a `gate_fire` value meka never wrote), it cannot be converted, because there is nothing to convert it *from*. Such a job never fired under 0.42, and it does not fire under 0.43 either: the migration leaves it in the same refused state rather than guessing at what it meant or deleting it. It is logged once, by id, at `warn`.

The consequence is that the row stays inert and invisible, as it already was: it will not appear in `meka schedule list` and `meka schedule cancel` cannot reach it. Recreate the job if you still want it. The original row is in the backup 0.43 took, `meka.db.v1.bak`. Note that from 0.44 a later schema-changing upgrade deletes that file, so put a copy somewhere of your own if you want to keep it.

## 0.41 to 0.42

A store written by 0.41 needs five conversions before 0.42 reads all of it. They are performed by `migrate-0.41-to-0.42.py`, a one-shot script attached as an asset to the 0.42 release. Download it, run it once, and you are done with it.

This one stays a script, and 0.43's own store migration does not replace it: 0.42 carried no migration code to reach back with, and conversion B below has to *guess*. 0.41 recorded nothing about which provider a thinking block came from, so the script tells them apart by the shape of the blob, and it reports what it read before it writes. A guess wants a human reading the counts, which is the one thing a migration that runs on every start cannot offer.

### Order

1. **Run 0.41 once, before you replace it.** It brings a store from an older release fully up to date; 0.42 carries no migration code and cannot.
2. **Install 0.42 and launch it once.** This is what creates the tables the script writes into, so it is not an arbitrary step you can move: run the script against a store that predates 0.42 and it stops with an explanation rather than guessing.
3. **Run the script**, first as a dry run, then with `--apply`.

```bash
python3 migrate-0.41-to-0.42.py            # reports what it would change; writes nothing
python3 migrate-0.41-to-0.42.py --apply    # does it
```

Read the dry run before you apply it. Conversion B in particular reports how many thinking blocks it read as Claude's and how many as OpenAI's, and 0.41 did not record which was which. If those counts do not match the providers you actually used, stop: the blocks it could not place are left alone, but the ones it places wrongly are not recoverable from the row afterwards.

The dry run is the only place to read that. Its per-class counts and its warning about a session holding both kinds describe the write it is about to do, so once the blocks are converted a later run has nothing left to report about them.

Between steps 2 and 3 the store is live but incomplete: memories are absent from the agent's index, and any session affected by conversion E below is already broken. Step 3 is part of the upgrade rather than cleanup to get to later.

The script finds meka's own directories by default, honoring `MEKA_CONFIG_DIR` and `MEKA_DATA_DIR`; `--root`, `--skills-root` and `--database` point it at a copy instead. `--self-test` checks the script against its own fixtures and exits, touching nothing of yours.

### What it converts

| Conversion | What it changes | If you skip it |
|---|---|---|
| **A.** Memories | The Markdown files under `<config>/memory/` become rows in the store's `memories` table, which is where 0.42 reads memories from. The files are read, never written or deleted. | The memories are simply not there. The files are untouched on disk, so nothing is lost and the import still works whenever you get to it. |
| **B.** Thinking blocks | A stored block's bare `signature` becomes an `opaque` object naming which provider it belongs to: `signed` for a Claude signature, `sealed` for OpenAI's encrypted reasoning. 0.41 wrote both to the same field and recorded nothing about which was which, so the script tells them apart by the shape of the blob and **reports the counts before it writes**. A blob it does not recognize is left exactly as it is. | The block loses its opaque half, so that reasoning stops being replayed to the provider. The session still loads and still runs; it just resumes without the chain of thought behind those turns. |
| **C.** A skill's `version:` / `author:` | Both move from the top level of a `SKILL.md`'s frontmatter under `metadata:`, keeping their names, which is where the Agent Skills spec puts them. | Nothing. meka reads a top-level `version:` and `author:` permanently, because Claude Code's plugin skills declare `version:` there. This conversion is cosmetic. |
| **D.** A skill's `priority:` | Moves under `metadata:` **and is renamed** to `meka-priority:`. | The skill silently drops to the default rank of 5. A rank is read from `metadata.meka-priority` and nowhere else, so the `[Skills]` index comes out in a different order and its cap drops different skills. Nothing warns. |
| **E.** A stored `tool_result` | Content held as a bare JSON string becomes a list of typed blocks, `[{"type": "text", "text": ...}]`. | The affected session breaks. The row will not deserialize, so it is dropped as the session loads, which orphans the `tool_use` it answered, and the provider then refuses the next turn. |

### The two that matter

A and B announce themselves: a memory you saved is missing from the index, or a thinking block is not replayed. Both are recoverable by running the script later.

**D and E are the ones that damage silently.** D changes which skills the `[Skills]` index shows first and which its cap drops, with nothing on screen to say the rank it used was not the one in your file. **E can leave a session unusable**: it loads cleanly, and then the next turn is refused by the provider because a `tool_use` in the history has no matching result. See [Sessions](../usage/sessions.md#rewinding-a-session) if you have already met that error.
