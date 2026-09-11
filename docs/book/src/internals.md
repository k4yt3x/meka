# Internals

A map of the source tree for contributors: how the modules depend on each other, how one turn
travels through them, where state lives, which rules are enforced by a single named predicate, and
what each host offers. The code is the reference; this page says where to look. Every name here
was checked against the tree when it was written, so if a name is missing, `grep` for it before
assuming the page is right.

## Layering

The tree reads top-down. Each module calls the ones below it and never the ones above or beside it,
and `tests/layering.rs` scans every `crate::` path in production code to keep it that way. Every
top-level module has a rank there; an edge may only point at a strictly greater rank. The tolerated
list is empty. One upward edge is by design, `tools/subagent.rs` building an `Agent`, because a
sub-agent is an agent.

```text
rank  module          holds
 0    main            argument parsing; picks a host or a cli handler
 1    host            the layer between a host and the agent, and the four hosts under it
 2    cli             clap definitions and one handler file per subcommand group
 2    relay           tracing output routed around the live REPL prompt
 3    console         the terminal between two prompts: spacing, notices, errors
 4    render          markdown, tool indicators, todo lists, status lines
 5    agent           the turn loop, tool dispatch, compaction, recovery
 5    view            the JSON record shapes --format json and the HTTP API share
 6    tools           the Tool trait, the registry, the built-ins, MCP tools as Tools
 7    prompt          the system prompt and the per-turn context block
 8    session         the materials a session is built from, and its live cells
 9    scheduler       the sweep that claims due jobs and hands a Wakeup to a host
 9    background      tool calls the agent starts and does not wait for
10    mcp             the MCP client: connections, published tools, progress, auth
11    provider        the Provider trait, wire types, the registry, one backend per API
12    frontend        the Frontend trait, its events, and the two generic frontends
12    skills          skill discovery and loading
12    instructions    the standing instructions file
12    oauth           expiry, refresh, the refresh lock, PKCE
12    sandbox         read-only confinement for execute_command
13    workspace       cwd, roots, the write fence, private-directory refusals
13    tokens          token estimates for the gauge between provider reports
14    store           every SQL statement, behind one connection owner
15    schedule        what a scheduled job is: parsing, gates, the scheduler's memory
16    conversation    the event-log conversation and its title
16    stats           per-session counters
16    config          config.toml, ResolvedConfig; profile.rs holds accounts and profiles
16    memory          the memory entry
17    entry           what skills and memories share: an indexed entry
17    permission      levels, the enabled set, the shared cell, the approvals switch
17    todo            the task list
17    image           format detection and transcoding
18    fs              private directories, atomic replace, file locks
19    error           MekaError
19    sync            locks that outlive a panic
19    streams         raw stderr
19    paths           the config and data directories
20    text            pure text helpers: widths, columns, sizes, timestamps, unknown_name
```

`relay` is installed by the two terminal hosts (`host/repl.rs`, `host/oneshot.rs`) and by
`main.rs` as tracing's writer; it writes through `console`, which draws with `render`. `render` and
`console` are reached only from `host`, `cli` and `relay`. `text` is the lowest leaf so that `error`
can render `MekaError::ProfileNotConfigured` through `text::unknown_name`.

Put new code where its callers already are: a type read by `store` and `host` belongs in `store` or
below, never in `host`. A new module fails the test until it is given a rank.

| Module | Holds |
|--------|-------|
| `src/main.rs` | Dispatch only: parse arguments, pick a host or a CLI handler, map the exit code. |
| `src/cli.rs`, `src/cli/` | The clap definitions and one handler file per subcommand group: `account`, `profile`, `session`, `history`, `mcp`, `tool`, `skills`, `memory`, `instructions`, `schedule`, `background`. Everything that owns the stdout/stderr contract lives here. |
| `src/host.rs`, `src/host/` | `host/assembly.rs` builds a session: `SharedDeps`, `build_session_agent`, `hydrate_conversation`, `resolve_profile_switch`, `record_session_change`. `host/session.rs` runs one: `ResidentSession`, `TurnGuard`, `BusyGuard`, the `CancelCell`, the `Sessions` registry and its idle sweep, `fork_and_lock`, outcome claiming. `host/scheduler.rs` is `HostHooks` and `run_wakeup`, the out-of-band turn every host runs the same way. `host/terminal.rs` is what the REPL and one-shot share: Ctrl+C, the interruptible turn. `host.rs` keeps `COMMANDS`, the slash-command table the REPL offers and ACP advertises the `for_editors` rows of. `host/repl`, `host/oneshot`, `host/acp` and `host/http` are the hosts. |
| `src/agent.rs`, `src/agent/` | The `Agent` and its options; `turn.rs` runs a turn and owns `TurnInput`, `dispatch.rs` executes tool calls and owns `admit_tool_call`, `compaction.rs` summarizes, `recovery.rs` decides what a failed request becomes. |
| `src/view.rs` | The record shapes `--format json` prints and the HTTP API serves, each defined once with its `From` conversion from the store or config type it shows: `SessionView`, `ProfileView`, `AccountView`, `McpServerView`, `McpToolView`, `ScheduledJobView`, `GateView`, `MemoryDetail`, `ToolView`, `SkillView`, `SkillDetail`. A host adds what only it can answer around the shared core by `#[serde(flatten)]`: the HTTP `SessionResponse` flattens `SessionView` under `last_turn_at`, `capabilities` and `turn_in_flight`; the CLI's `InstalledSkillView` and `ConfiguredToolView` flatten a core under what only a terminal should see, such as a path on this machine. Every `Option` field is omitted when absent, never `null`. |
| `src/session.rs` | What a session is made of: `CoreMaterials` and `SessionMaterials` (what every agent and registry of a session is built from), `SessionCells` (permission, cwd, roots, session id, todo list, the published profile, the context gauge, background tasks, the frontend, the session lock slot, a pending compaction), `ToolSite` (the four cells a built-in reads), `AgentOptions` and `CompactRequest`. No host and no SQL. |
| `src/provider.rs`, `src/provider/` | The `Provider` trait, the wire types, `MessageAccumulator`, the registry of profiles, and one backend per API: `anthropic/messages.rs` and `anthropic/subscription.rs` over `anthropic/shared.rs`; `openai/chat_completions.rs`, `openai/responses.rs` and `openai/subscription.rs`, the last two over `openai/responses_wire.rs`. Every streaming backend reads SSE through `provider/sse.rs`; the refresh-once rule for a rejected subscription credential is `oauth::send_with_one_refresh`; `provider/mock.rs` is the scripted provider the test suites drive. |
| `src/tools.rs`, `src/tools/` | The `Tool` trait, `ToolContext`, `admit_arguments`, the registry and its builders, the gate toolset, the built-in tools, and `mcp_adapter.rs`: each remote MCP tool as a `Tool`, and the registry as a subscriber to the client's tool lists. A built-in reads the session it serves through one `ToolSite`. |
| `src/mcp.rs`, `src/mcp/` | The MCP client: `ServerEntry` per server, the connector and its reconnects, the client handler, each server's tools published to whoever subscribes, progress and resource-update routing on `McpClientContext`, and auth. It never names a registry and never talks to a terminal: an interactive login goes through the `LoginPrompt` that only `meka mcp login` installs. |
| `src/store.rs`, `src/store/` | The store. `migrations.rs` is the append-only ledger; `locks.rs` the per-session file claim; `backup.rs` the copy taken before a migration; `export.rs` the archive format; every other file owns the statements for one table family. |
| `src/schedule.rs`, `src/schedule/` | Scheduled jobs as a domain: parsing, gates, and `SchedulerMemory` (`schedule/memory.rs`), the per-process record of which refusals have been reported. No SQL. `src/scheduler.rs`, above the store, is the sweep that fires them. |
| `src/config.rs`, `src/config/profile.rs` | `ConfigFile`, loading, environment substitution, `ResolvedConfig`, and the vocabulary every layer reads. `profile.rs` is `Backend`, an account and a profile as written, a profile as resolved through its account, `select_profile` and `require_profile`. Live services and probes are resolved by the host, not here. |
| `src/prompt.rs` | The system prompt and `build_turn_context`, the block the model sees ahead of every user turn. |
| `src/fs.rs`, `src/oauth.rs`, `src/sync.rs`, `src/text.rs` | Private directories and the file locks; PKCE, refresh and the loopback callback; the one place a poisoned lock is recovered; and every pure text rule the surfaces share. |

## One turn

A host admits the turn, the agent runs it, and everything the user sees comes back through a
`Frontend`.

1. **Admission.** A `ResidentSession` counts its work in one cell, `in_flight`, and holding the
   conversation mutex is what "a turn is in flight" means. HTTP and ACP admit a typed prompt with
   `admit_turn`, which hands back a `TurnGuard` and samples the cancel epoch; HTTP wraps it in
   `host::http::state::admit_turn` to add the process-wide cap. A scheduled fire or an outcome
   delivery takes `mark_busy`; HTTP compact and rewind take `claim_idle`, which refuses while
   anything is in flight. The REPL and the one-shot drive one session from one thread, so they
   sample `cancel.admit()` themselves and take the conversation lock.
2. **Input.** The host builds a `TurnInput`: the typed prompt or the outcomes it carries, images,
   and the retention, which the scheduler sets per job and the HTTP turn takes from the request.
   `TurnInput::from_parts` is the empty-prompt rule, raising
   `MekaError::EmptyPrompt` before admission.
3. **The loop.** `Agent::run_turn` appends one user message of two blocks: a `TurnContext` block
   holding everything meka injected (permission and environment context, todos, world state, budget,
   background outcomes, the resume notice) and a `Text` block holding the words as typed. Providers
   render the first as text ahead of the second; exports, `GET /messages`, replays and the title
   read the words. It then sends a `CompletionRequest` and dispatches every tool call the response
   carries. `admit_arguments` type-checks the call's arguments and takes out `background`, the flag
   that detaches it; `admit_tool_call` decides run, ask or refuse. Each call gets a `ToolContext`:
   the session id, the tool-use id, the prompt id, the frontend and the cancellation token.
4. **Recovery.** A failed request goes through `TurnRecovery`, which decides between a retry, a
   degraded resend and a reported failure. Compaction runs when the context gauge says so, when the
   model asks through `context_compact`, or when the user asks.
5. **Output.** Text, thinking, tool indicators, approval prompts and elicitations all reach the
   user through the session's `Frontend`. The REPL, ACP and HTTP each implement it once; a
   sub-agent's `PermissionForwardingFrontend` forwards its approval prompts and notices to its
   parent's; `SilentFrontend` answers a call with nobody behind it.

## Where state lives

Non-secret settings live in `config.toml`. Secrets live in the store. Environment variables are
operational only. The store is one SQLite file, `meka.db` under `MEKA_DATA_DIR`, opened by `Store`,
and its shape is whatever the migration ledger says it is: nineteen entries today, so a current store
reads `PRAGMA user_version = 19`, and `HEAD_SCHEMA_FINGERPRINT` in `store/migrations.rs` pins the
columns of the eleven tables in `HEAD_TABLES`. The last entry, `background_tasks_spell_canceled_with_one_l`, rewrites a task status an earlier meka stored as `cancelled` to the `canceled` the reader accepts. Before it, `root_rows_take_the_default_level_once_the_config_reads` stamps `[permissions].default` on a root row that still records no level and refuses to migrate while `config.toml` cannot be read, so the store keeps its shape until the file reads.

| Table | Owner | Columns and indexes |
|-------|-------|---------------------|
| `sessions` | `store/sessions.rs` | `id`, `created_at`, `updated_at`, `parent_session_id`, `cwd`, `permission`, `approvals`, `profile`, `capabilities_json`, `token_id`, `additional_roots_json`, `subagent_spec_json`, the `stat_*` counters. `idx_sessions_updated_at`, `idx_sessions_parent_session_id`. |
| `messages` | `store/sessions.rs` | `id`, `session_id`, `role`, `content`, `created_at`. `idx_messages_session_id`. |
| `account_credentials` | `store/credentials.rs` | `account`, `credentials_json`, `updated_at`. The `credentials_belong_to_accounts` step leaves a `provider_credentials` view over it, not for any reader but for the frozen `sessions_name_their_provider` step, which queries that name and would fail a replay on a store that lost its `user_version`. |
| `mcp_credentials` | `store/credentials.rs` | `server_name`, `kind`, `secret`, `updated_at`; keyed by `(server_name, kind)`, so a client secret and its refreshable bundle coexist. |
| `blobs`, `message_blobs` | `store/blobs.rs` | `hash`, `media_type`, `bytes`, `size`, `created_at`; and `message_id`, `hash` for which message rows reference a blob. `idx_message_blobs_hash`. |
| `tool_outputs` | `store/scratchpad.rs` | Scratchpad entries: `session_id`, `name`, `content`, `created_at`. The table keeps its old name. |
| `scheduled_jobs` | `store/schedule.rs` | `id`, `session_id`, `kind`, `spec`, `prompt`, `gate_kind`, `gate_spec_json`, `gate_last_output`, `gate_permission`, `claimed_by`, `claim_expires_at`, `attempts`, `created_at`, `last_fired_at`, `next_fire_at`. `idx_scheduled_jobs_next_fire_at`, `idx_scheduled_jobs_session_id`. |
| `background_tasks` | `store/background.rs` | `id`, `session_id`, `tool_name`, `label`, `status`, `outcome`, `scratchpad_name`, `started_at`, `finished_at`, `announced_at`, `delivered_at`. `idx_background_tasks_session_status`. |
| `memories`, `memories_fts` | `store/memory.rs` | `id`, `name`, `description`, `tags`, `body`, `priority`, `created_at`, `updated_at`, `read_count`, `last_read_at`. `idx_memories_rank`. The FTS index and its triggers are rebuilt to this build's definition by `reconcile_index` on every open, so they are outside the ledger's fingerprint. |
| `prompt_history` | `store/history.rs` | The REPL's input history: `id`, `command_line`, `created_at`. |

Per-process state that is not a table has an owner too: the scheduler's memory of which refusals it
has already reported is the `SchedulerMemory` on the `Store`; MCP progress routing and resource
updates live on the `McpClientContext`; the per-path write locks are the `WriteLocks` on the host's
`SharedDeps`, cloned into every session's `CoreMaterials`; a frontend's sticky approval answers are
its `StickyApprovals`, in memory and never persisted.

## The doors

Most defects that survive review are one rule enforced at one entry point and forgotten at a
sibling. The rules below are each written once and called from every path that reaches the thing
they guard. When adding a path, call the predicate rather than restating the rule.

| Rule | Predicate | Doors |
|------|-----------|-------|
| Whether a turn may start | `ResidentSession::admit_turn`, `mark_busy`, `claim_idle` on the one `in_flight` cell | `admit_turn`: HTTP `POST /turn` (under the process cap, `host::http::state::admit_turn`) and ACP `session/prompt`; `mark_busy`: scheduled fires and outcome deliveries in `host/scheduler.rs`; `claim_idle`: HTTP compact and rewind. The REPL and one-shot sample `cancel.admit()` directly. |
| Whether a second turn is refused | the conversation mutex, `try_lock` | HTTP `POST /turn` (`try_lock_owned`, 409), ACP `session/prompt` (`InvalidParams`) |
| Whether a session is idle enough to evict | `host::session::idle_on`, behind `ResidentSession::is_idle_given` and `Sessions::sweep_idle` | the HTTP GC and the ACP idle sweep |
| Whether a prompt is empty | `TurnInput::from_parts` | every host, before admission |
| Whether a tool call runs, is put to the user, or is refused | `admit_tool_call` in `agent/dispatch.rs` | inline dispatch, the checkpoint turn in `agent/compaction.rs` |
| What a tool call's arguments are, and whether it detaches | `tools::admit_arguments` | dispatch, ahead of every tool |
| What level a scheduled job runs at | `scheduler::live_permission`, reading the session row through `admit_recorded` | the fire door, the wake watcher, `meka schedule show` |
| Whether a recorded level still applies | `EnabledPermissions::admit_recorded` | a resume on every host (`host/assembly.rs`, `host/http/reattach.rs`, `host/acp/session.rs`), the scheduler |
| The refusal for a level the set does not admit | `EnabledPermissions::disabled_level`, raising `MekaError::DisabledLevel` | `SharedPermission::try_set` behind `/permission`, ACP `session/set_mode` and its permission option; HTTP create and `PATCH` |
| Whether a profile name is configured | `config::require_profile`, raising `MekaError::ProfileNotConfigured` | `--profile` and `default_profile` selection, `resolve_profile_switch` and the resume repin (`host/assembly.rs`), HTTP create and `PATCH`, the provider registry, `meka profile use` |
| What a session's row may move to, and what a failed write costs | `Store::update_session` taking a `SessionPatch`; the policy is `host::record_session_change` | REPL `/permission`, `/approvals`, `/cd`; ACP `session/set_mode` and `session/set_config_option`; HTTP `PATCH`; the profile switch and resume reconciliation. A patch that fails to write the level, switch, directory or roots is warned about and the command continues; one carrying a profile fails the request. The one writer below `host` is the sub-agent tool recording the profile a worker ran on, which warns. |
| Whether a working directory is accepted, and how it is spelled | `workspace::accept_cwd`; `cwd_filter` for a listing filter | ACP `session/new`, `load`, `resume`, `fork`; REPL `/cd`; HTTP create, `PATCH` and `?cwd=` |
| Whether a session accepts an image | `ResidentSession::accepts_images`, off the published profile's `vision` | HTTP `POST /turn`, ACP `session/prompt` |
| A session's title | `Conversation::title` (the first user `Append` in the log with a non-blank `Text` block that is not a redaction placeholder or harness note, whitespace collapsed, `TITLE_CHARS` = 80; `TITLE_ROW_WHERE_SQL` is the same rule in SQL) | `SessionSummary`, the HTTP turn response, ACP's title update |
| Making a persisted session resident | `Store::open_session_row`: lock first, then read | REPL and one-shot resume, `serve` re-attach (`ensure_session_loaded`), ACP `session/load` and `session/resume` |
| Forking | `Store::fork_session_locked` with `SourceLock::{Probe, HeldByCaller}` | `meka session fork` (`Probe`); REPL `/fork` through `host::fork_and_lock` (`HeldByCaller`); HTTP `POST /fork` and ACP `session/fork`, which hold a resident source still under `HeldByCaller` (refusing it mid-turn) and `Probe` a dormant one |
| What an HTTP caller may do | `scope::Scoped<R>` as an extractor | every handler, by its signature |
| Whether a path may be written | `workspace::WriteScope` | `write_file`, `edit_file`, `scratchpad_save_file`, `execute_command`'s confinement |
| Whether meka's own directories may be read | `workspace::private_read_refusal` and `resolves_into_private` | `tools/util.rs` for the readers, `find_files`, `search_contents` |
| Whether a backend reads a profile or account key | `Backend::reads_profile_key` and `reads_account_key` (in `config.rs`) | `profile add`, `profile set`, `account add`, the load-time warning |
| Which account a profile bills | `config::account_for` | config validation, the provider registry |
| Whether an MCP server's config may be sent at all | `ServerEntry::refused`, a field set at construction | every connect door in `mcp/connector.rs`, `ServerEntry::reconnect` |
| Whether a fire's session is still resident | `HostHooks::still_resident` | scheduled fires and outcome deliveries, once the lock is won |
| What a frontend answered for a file operation | `Delegation` | `read_file`, `edit_file`, `write_file` |
| Whether thinking is on for a request | `ThinkingOverride` on `CompletionRequest` | the turn, the summarizer, the checkpoint turn |
| Where an image's bytes rest | `store/blobs.rs`: `externalize_images` on write, `inline_blobs` on read | writes: `save_event`, `save_events_atomic`, `import_sessions`, the fork's row copy; reads: `host::hydrate_conversation`, the sub-agent follow-up; `GET /messages` serves the reference, `GET /blobs/{hash}` the bytes, and an export carries both |
| The sentence for a name that matches nothing | `text::unknown_name` | every refusal by name: profiles, accounts, MCP servers, configuration options, scopes, gates |
| How a terminal shows a time, a size, an id | `text::format_timestamp` with `Precision`, `text::format_size` with `KIB` and `MIB`, `text::ID_PREFIX` | every listing and status line; the wire keeps RFC 3339 and raw byte counts |
| The `Authorization` value for a token | `text::bearer` | every backend that sends one, and `store/credentials.rs` |
| What a sticky approval answer covers | `StickyApprovals` in `frontend.rs` | the REPL, ACP and HTTP frontends |
| How long a host waits for an approval answer | `APPROVAL_TIMEOUT` in `frontend.rs` (30 minutes) | the ACP and HTTP frontends; the REPL has a human and no timeout |
| How a notice serializes | `NoticeView` | SSE `notice`, the blocking response, the one-shot JSON report |
| Whether an upstream's own words may reach a caller | `host::relay_provider_errors`, bounded by `error::bounded_upstream_body` | `ProblemDetail::for_error`'s `provider_response`, `acp_error_for`'s `data` |
| Whether a skill's name resolves outside meka's own store | `skills::refuse_foreign_write` / `refuse_foreign_delete`, both on `foreign_location` | `skill_write`, `skill_delete`, `meka skill add`/`remove`, `PUT`/`DELETE /v1/skills/{name}`; the `ForeignSkill` it hands back renders with the path for a local reader and without it on the wire |
| Whether an older meka wrote the store | `store/migrations.rs` alone | nothing else may know |

### One error, one mapping per host

`MekaError` carries the refusals the doors raise: `SessionNotFound`, `TurnInFlight { doing }`,
`EmptyPrompt`, `DisabledLevel { level, enabled }`, `ProfileNotConfigured { name, known }`,
`RequestTooLarge`, beside `SessionLocked`, `SessionNotDrivable`, `Usage` and `Config`. Each host
maps the enum once, and no handler rewrites the sentence:

- HTTP: `ProblemDetail::for_error` in `host/http/errors.rs`. `SessionNotFound` is 404
  (`session-not-found`); `TurnInFlight` and `SessionLocked` are 409 (`turn-in-flight`,
  `session-locked`); `EmptyPrompt`, `DisabledLevel`, `ProfileNotConfigured`, `Usage` and `Config`
  are all 422 under `invalid-body`, so a client tells them apart by `detail`; `RequestTooLarge` is
  422 under its own `request-too-large`, because meka refused it and there is no provider response
  behind it. The `type` URIs live under `https://meka.so/errors/`.
- ACP: `acp_error_for` in `host/acp.rs`. Everything the caller can act on is `InvalidParams`
  (`-32602`) with the `Display` as `data`; everything else is `InternalError`.
- REPL: `console.error(&error)`, the `Display` on stderr.
- One-shot and CLI: the `Display` on stderr and exit code 1; 130 when the turn was interrupted.

**Two classes never travel verbatim, on either wire.** `Installation` is meka's own sentence about
the *operator's* setup -- a `[web]` client that will not build, a `base_url` shape a backend refuses
-- so it names a path or an endpoint from `config.toml` that no caller can act on: HTTP answers a
sanitized 500 and ACP an `InternalError` with the text in the log, while the REPL and the CLI print
it, since their reader is the operator. The web client is built once in `build_shared_deps`, so on
every host a bad `[web]` block fails the process at startup rather than surfacing per session.

And an upstream's own response text travels only when `[serve] relay_provider_errors` says so, which
`host::relay_provider_errors` reads once for both hosts. `for_error` attaches it as
`provider_response`; `acp_error_for` appends it to `data`; both bound it with
`error::bounded_upstream_body`. An MCP connector's reason, a `Database` and an `Io` never travel at
all, on either.
- REPL: `console.error(&error)`, the `Display` on stderr.
- One-shot and CLI: the `Display` on stderr and exit code 1; 130 when the turn was interrupted.

## What each host offers

The hosts do not offer the same operations, and the gaps are recorded here rather than filled.
"Run" is `meka` with `-p`, `-c`, `-r`, `--profile` and `--permission`; the one-shot is that run
with `--oneshot`.

| Operation | REPL | One-shot | ACP | HTTP | `meka session` |
|-----------|------|----------|-----|------|----------------|
| New session | at launch | at launch | `session/new` | `POST /v1/sessions` | no |
| Resume | `-c`, `-r` at launch | `-c`, `-r` at launch | `session/load`, `session/resume` | implicit re-attach of a dormant id on any request | no |
| Fork | `/fork` | no | `session/fork` | `POST /{id}/fork` | `fork` |
| Delete | no | no | no (`session/close` releases only) | `DELETE /{id}` | `delete` |
| Rewind | `/rewind` | no | no | `POST /{id}/rewind` | `rewind` |
| Compact | `/compact` | automatic only | automatic only | `POST /{id}/compact` | no |
| Export | `/export` (Markdown) | no | no | `GET /{id}/export` | `export` |
| Import | no | no | no | `POST /v1/sessions/import` | `import` |
| Profile switch | `/profile` | `--profile` at launch | `session/set_config_option` | `PATCH` `profile` | no; `meka -r --profile` repins |
| Level switch | `/permission`, Shift+Tab | `--permission` at launch | `session/set_mode`, config option | `PATCH` `permission` | no |
| Approvals | `/approvals` | config only | `session/set_config_option` | `PATCH` `approvals` | no |
| List, show | `/session`, `/status` | no | `session/list` | `GET`, `GET /{id}` | `list`, `show` |
| Cancel a turn | Ctrl+C | Ctrl+C | `session/cancel` | `POST /{id}/cancel` | no |

The REPL and the one-shot share `COMMANDS` in `host.rs` for what a slash command is; ACP advertises
the three rows marked `for_editors` (`/mcp`, `/status`, `/usage`) as `available_commands`.

### Mid-turn

Only HTTP and ACP can receive a request while a turn holds the session, and the two answer
differently by design.

- **HTTP refuses.** `PATCH`, `DELETE`, fork, compact and rewind check `in_flight` (or fail
  `claim_idle`) and return 409 with `type` `https://meka.so/errors/turn-in-flight`, through
  `turn_in_flight_conflict` and `ProblemDetail::for_error(MekaError::TurnInFlight)`. A second
  `POST /turn` on the session gets the same 409. The `detail` names what was refused (`doing`), and
  a `session_id` member carries the id.
- **ACP applies the level and the switch, and refuses the rest.** `session/set_mode` and the
  permission and approvals options of `session/set_config_option` write the cells without taking the
  conversation mutex, so an editor toggle takes effect on the very next tool call, and then record
  the row. The profile option `try_lock`s the conversation first and refuses with `InvalidParams`
  (`TurnInFlight { doing: "switch profile" }`) so that a switch refused for a turn in flight leaves
  the row where it was. A second `session/prompt` is refused `InvalidParams`
  (`TurnInFlight { doing: "prompt" }`), and so is a `session/fork` of the session
  (`TurnInFlight { doing: "fork the session" }`), which `try_lock`s the conversation the same way
  and holds it across the copy.

## Frontends

Six things implement or stand in for `Frontend`: `ReplFrontend` (`host/repl/frontend.rs`, also the
plain one-shot), `AcpFrontend` (`host/acp/frontend.rs`), `HttpFrontend`
(`host/http/http_frontend.rs`, which records for the blocking response and broadcasts to SSE
through `host/http/sse.rs`), `JsonFrontend` (`host/oneshot.rs`, `--format json`), `SilentFrontend`
and `PermissionForwardingFrontend` (`frontend.rs`; the latter forwards notices and approval prompts
to the parent's frontend and drops the rest). Where they differ:

| Event | REPL | ACP | HTTP stream | HTTP blocking | One-shot JSON | Silent |
|-------|------|-----|-------------|---------------|---------------|--------|
| `Notice` info / warn | `console.notice`, dim or warn-colored | agent-message chunk prefixed `[meka]` / `[meka warn]` | `notice` event (`NoticeView`) | `notices[]` | `notices[]` | dropped |
| `McpProgress` | inline status line | `tracing::info!` | `progress` event | dropped | dropped | dropped |
| `Compacted` | nothing (`/compact` prints `render::compaction_summary`) | info notice | `context.compacted` event | dropped; `GET /messages` carries the marker | dropped | dropped |
| Approval with nobody to ask | warn `approval_refused_without_asking`, deny (REPL thread gone) | asks the client; deny after `APPROVAL_TIMEOUT`, `Canceled` on cancel | `permission_required` event; deny after `APPROVAL_TIMEOUT` or on disconnect | warn notice in its own words (`stream=false has no channel`), deny | warn `approval_refused_without_asking`, deny | deny; the notice goes nowhere |
| Elicitation | asks through the REPL thread; warn `elicitation_declined` and decline when it is gone | `elicitation/create`; warn `elicitation_declined` and decline when the client lacks the mode | warn `elicitation_declined`, decline | same | trait default: warn `elicitation_declined`, decline | same, dropped |
| Scheduled fire prompt | dim info notice on the console | `UserMessageChunk` | info notice into the stream | info notice, drained after the turn | no scheduler | n/a |
| Scheduled fire failure | `console.error`; "interrupted" annotation on a cancel | warn notice `scheduled job '<id>' failed: ...`; info on a cancel | `schedule.fired` webhook, `status` `completed`, `canceled` or `failed`; nothing on the frontend | same webhook | no scheduler | n/a |

The scheduled-fire rows come from each host's `HostHooks` (`show_prompt`, `finished`) rather than
its `Frontend`, in `host/repl.rs`, `host/acp/schedule.rs` and `host/http/schedule.rs`.

## Building

Two Cargo features shape a build. `serve`, on by default, is `meka serve`: the HTTP API and the
dependencies only it needs, so `--no-default-features` builds a meka without one; `src/host/http.rs`
is behind `cfg(feature = "serve")`. `mock-provider` compiles in `provider/mock.rs`, the scripted
provider the test suites drive every host with; debug builds carry it regardless, and CI enables it
so a release-profile build is testable too. A shipped artifact is built without it. At run time,
`MEKA_MOCK_PROVIDER=1` selects that provider on every host (`provider/registry.rs`,
`host/assembly.rs`) and `MEKA_MOCK_PROVIDER_SCRIPT` names the JSON script it plays back.

The integration crates are gated on the same fact: `tests/acp.rs` and `tests/cli.rs` carry
`#![cfg(any(debug_assertions, feature = "mock-provider"))]`; `tests/serve.rs` and
`tests/multiprocess.rs` add `feature = "serve"`; `tests/repl_pty.rs` adds `unix`. Only
`tests/layering.rs` runs unconditionally. Unit tests open the store with `Store::for_test()`, an
in-memory database at the current schema; `Path::new(":memory:")` is spelled out because `None`
means the default path. The integration crates share `tests/harness/support.rs`, whose `Install` is
a temporary root with a config directory, a data directory and a work directory; `Install::env`
points a `meka` command at it (`MEKA_CONFIG_DIR`, `MEKA_DATA_DIR`, `HOME`, `XDG_*`,
`MEKA_MOCK_PROVIDER=1`, the script when one was written) and `Install::meka(args)` builds one.

The exact gate CI runs is in `AGENTS.md` under "Build gate"; run it before declaring a change done,
since clippy and rustdoc deny warnings there and not locally. The one-shot conversion for a 0.45
`config.toml`, `migrate-0.45-to-0.46.py`, is not in the repository: like the 0.42 script before it,
it is attached to its release as an asset by hand and carries its own `--self-test` fixture.
