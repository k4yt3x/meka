# Tools overview

Tools are the actions that the agent can perform on your behalf. The LLM decides which tools to call based on your instructions.

## Available tools

| Tool | Permission | Description |
|------|-----------|-------------|
| [`file_read`](./file-operations.md#file_read) | Read | Read file contents |
| [`file_edit`](./file-operations.md#file_edit) | Workspace | Make string replacements in a file |
| [`file_write`](./file-operations.md#file_write) | Workspace | Create or overwrite a file |
| [`file_find`](./search.md#file_find) | Read | Find files by glob pattern |
| [`file_search`](./search.md#file_search) | Read | Search file contents with regex |
| [`web_fetch`](./web.md#web_fetch) | Read | Fetch a web page as markdown |
| [`shell_execute`](./shell.md#shell_execute) | Read | Run a shell command (see the note below) |
| [`todo_write`](./overview.md#todo_write-todo_edit-todo_read) | Read | Create or replace the task list |
| [`todo_edit`](./overview.md#todo_write-todo_edit-todo_read) | Read | Update task statuses by number |
| [`todo_read`](./overview.md#todo_write-todo_edit-todo_read) | Read | Read the task list |
| [`agent_spawn`](./overview.md#agent_spawn) | Read | Delegate tasks to a sub-agent |
| [`agent_list`](./overview.md#agent_list--agent_followup--agent_steer--agent_delete) | Read | List the sub-agents this session spawned |
| [`agent_followup`](./overview.md#agent_list--agent_followup--agent_steer--agent_delete) | Read | Ask a sub-agent another question |
| [`agent_steer`](./overview.md#agent_list--agent_followup--agent_steer--agent_delete) | Read | Send a sub-agent a message without waiting for it |
| [`agent_delete`](./overview.md#agent_list--agent_followup--agent_steer--agent_delete) | Read | Discard a sub-agent and its records |
| [`scratchpad_write`](./scratchpad.md#scratchpad_write) | Read | Store content in the scratchpad |
| [`scratchpad_read`](./scratchpad.md#scratchpad_read) | Read | Read a scratchpad entry |
| [`scratchpad_edit`](./scratchpad.md#scratchpad_edit) | Read | Edit a scratchpad entry |
| [`scratchpad_list`](./scratchpad.md#scratchpad_list) | Read | List scratchpad entries |
| [`scratchpad_delete`](./scratchpad.md#scratchpad_delete) | Read | Delete a scratchpad entry |
| [`scratchpad_merge`](./scratchpad.md#scratchpad_merge) | Read | Combine several scratchpad entries into one |
| [`scratchpad_rename`](./scratchpad.md#scratchpad_rename) | Read | Rename a scratchpad entry |
| [`scratchpad_load_file`](./scratchpad.md#scratchpad_load_file) | Read | Load a file into the scratchpad |
| [`scratchpad_save_file`](./scratchpad.md#scratchpad_save_file) | Workspace | Write a scratchpad entry out to a path |
| [`skill_read`](./overview.md#the-skill_-tools) | Read | Load a named skill's instructions |
| [`skill_search`](./overview.md#the-skill_-tools) | Read | Regex over the full text of every skill |
| [`skill_write`](./overview.md#the-skill_-tools) | Read | Create or update a skill |
| [`skill_delete`](./overview.md#the-skill_-tools) | Read | Delete a skill and its directory |
| [`memory_write`](../usage/memory.md) | Read | Save a durable note that outlives the session |
| [`memory_read`](../usage/memory.md) | Read | Read a saved memory, paging long bodies |
| [`memory_search`](../usage/memory.md) | Read | Ranked full-text search over every memory |
| [`memory_delete`](../usage/memory.md) | Read | Delete a saved memory |
| [`image_render`](./overview.md#image_render) | Read | View an image from in-memory base64 or scratchpad |
| [`context_check`](./overview.md#context_check--context_compact) | Read | Measure the context window live: occupancy, headroom, compaction count |
| [`context_compact`](./overview.md#context_check--context_compact) | Read | Ask for a compaction before the next step of this turn |
| [`conversation_search`](./overview.md#conversation_search--conversation_read) | Read | Search the full conversation history, including compacted turns |
| [`conversation_read`](./overview.md#conversation_search--conversation_read) | Read | Read conversation turns by index |
| [`schedule_create`](../usage/scheduling.md) | Read | Schedule a future turn for this session |
| [`schedule_list`](../usage/scheduling.md) | Read | List this session's scheduled jobs |
| [`schedule_cancel`](../usage/scheduling.md) | Read | Cancel a scheduled job |
| [`task_list`](../usage/background.md) | Read | List this session's background tasks |
| [`task_cancel`](../usage/background.md) | Read | Stop a running background task |
| [`tool_load`](./overview.md#deferred-tools) | Read | Fetch the full schema of a deferred tool, one name or up to ten |
| [`tool_search`](./overview.md#deferred-tools) | Read | Find a tool by keyword across names and descriptions, deferred ones included |

The `schedule_*` tools require [`[schedule] enabled`](../configuration/config-file.md#schedule) (on by default), the `memory_*` tools require [`[memory] enabled`](../configuration/config-file.md#memory) (on by default), and the `task_*` tools require [`[background] enabled`](../configuration/config-file.md#background) (off by default). `skill_write` and `skill_delete` require [`[skills] agent_managed`](../configuration/config-file.md#skills) (off by default) and are never given to a sub-agent. A disabled subsystem registers no tools at all, rather than shipping schemas that could only fail.

## Permission requirements

Tools are grouped by the minimum permission level required:

**Read permission** (available at `read` and above):
- `file_read`, `file_find`, `file_search`, `web_fetch`
- `shell_execute` (sandboxed, filesystem write-protected)
- `todo_read`, `todo_write`, `todo_edit`, `agent_spawn`, `agent_list`, `agent_followup`, `agent_steer`, `agent_delete`, `image_render`
- All skill tools, including `skill_write` and `skill_delete` when they are enabled: like memory,
  skills live in meka's own config directory, not your working tree
- `conversation_search`, `conversation_read`, `context_check`, `context_compact`
- Every scratchpad tool except `scratchpad_save_file`, which writes to a path you name and so sits at `workspace` with `file_write`
- All memory tools. Writing a memory needs only read permission: memories live in the store,
  which is meka's own, not your working tree.

**Workspace permission** (available at `workspace` and above; writes are confined to the workspace roots at `workspace`):
- `file_edit`, `file_write`, `scratchpad_save_file`

`shell_execute` is not in that list: it asks for `read` when a sandbox backend is available and `unrestricted` when none is, so it is reachable at `read` and confined by the *level*, not by its own requirement.

With [approvals](../usage/permissions.md#approvals) on, a call above the level is put to you instead of refused. An approved call still runs at the session's level: an approved `shell_execute` at `read` runs in the read-only sandbox, and an approved `file_write` lands only under the workspace roots. Raise the level when an approved call needs more reach.

At **none**, no tools are available. The agent can only respond with text.

## Filtering built-in tools

Any built-in can be allow-listed, blocked, or have its required permission overridden via the `[tools]` table in `config.toml`. See [`[tools]`: built-in tool filters](../configuration/config-file.md#tools-built-in-tool-filters). Run `meka tool list` to see every built-in with its effective permission and current status.

## MCP tools

When [MCP servers](../configuration/config-file.md#mcpservers) are configured, their tools are registered under a namespaced name of the form `mcp__<server>__<tool>` (e.g. `mcp__notion__notion-search`). The `mcp__` prefix matches [Claude Code](https://github.com/anthropics/claude-code)'s convention and keeps MCP tools from colliding with built-in names. They appear in the per-turn context catalog alongside the built-ins, with their resolved permission level annotated inline, and are called the same way.

meka also exposes seven built-in **MCP meta-tools** for browsing server-side resources and prompts. All are deferred by default; `tool_search` finds them, and `tool_load` with the exact name makes the schema available on the next model request within the same turn:

| Tool | Permission | Description |
|------|-----------|-------------|
| `mcp_resource_list` | Read | List resources a server exposes |
| `mcp_resource_read` | Read | Read a server resource by URI |
| `mcp_prompt_list` | Read | List server-defined prompts |
| `mcp_prompt_get` | Read | Render a server prompt with arguments |
| `mcp_resource_subscribe` | Read | Receive change notifications for a resource |
| `mcp_resource_unsubscribe` | Read | Stop receiving change notifications |
| `mcp_resource_updates_list` | Read | Inspect the latest recorded resource updates |

## Deferred tools

Most MCP tools are **deferred**: they are registered and listed under `[Tool discovery]` in the per-turn context, but their JSON schemas are withheld from the request until the agent calls `tool_load`. A large server can advertise fifty tools with multi-kilobyte schemas, and shipping all of them on every turn costs more than it returns.

The listing is bounded, like the skills and memory indexes. Every deferred tool appears under its server with the permission level it requires. A one-line summary rides along while the whole section fits 8 KB; past that every entry keeps its name and loses its summary, and past 200 names the section stops and counts what it left out. Each server heading carries its tool count in every tier and headings are kept while they fit, so a server whose entries were cut is still named with its size; past that, the count line says how many groups are not listed at all.

The trade-off is that until a tool is loaded, the agent sees at most its name and a summary clipped to 250 characters. **Anything past that clip is invisible**, including optional parameters, and a summary that was clipped ends in `…`.

Two behaviors exist so this never turns into a silent wrong answer:

- Calling a deferred tool without loading it **works**. The agent may be confident about the required arguments, and forcing a round trip it doesn't need is worse than allowing it.
- But when it does that and the tool has documented parameters it didn't pass, meka appends a note to the tool result naming them, with their types, defaults, and descriptions. A wrong default stops being invisible. The note is emitted once per tool per run.

`tool_search` finds a tool by keyword across every registered tool's name and full description, deferred ones included. Matching is forgiving: a word in the name outranks one in the description, a whole word outranks a substring, and a near miss (`xerfs` for `xrefs`) still counts. Each result carries the tool's summary, whether it is deferred, and whether the session's current permission level and approvals switch allow a call, by the same rule dispatch applies, so the agent never loads a tool only to find the call refused at that door. A tool's own refusals once called (a shell with no sandbox at `workspace`, a path outside the roots) depend on the arguments and are not predicted. Up to ten results are shown, which is one `tool_load` batch.

`tool_load` takes one name or an array of up to ten, so a task needing several tools off one server costs one round trip. Each loaded schema is headed by the same line on whether the level allows a call, and a name that matches nothing is answered with the nearest registered names or, failing that, the keyword matches for the word:

```text
tool_load({"name": ["mcp__notion__search", "mcp__notion__fetch"]})
```

Sub-agents see the same picture: a worker's system prompt carries the identical `[Tool discovery]` section over its own registry (its parent's servers minus whatever [`[subagents]`](../configuration/config-file.md#subagents) and `agent_spawn` deny), and it has `tool_search` and `tool_load` like the root.

Tools listed in a server's [`eager_load_tools`](../configuration/config-file.md#mcpservers) skip all of this: their schemas ship from turn 1. Use it for tools whose optional parameters matter and that the agent reaches for constantly.

**When writing a tool description for a server meka will consume**, put whatever a caller must know to use the tool correctly in the first two sentences. That may be all anyone ever sees.

## Background calls

With [`[background] enabled`](../configuration/config-file.md#background), every tool except `context_compact` gains an optional `background` parameter, MCP tools included. `context_compact` does no work of its own: it parks a request the loop drains once the batch's results are in, and detaching it would race that drain. A call that sets it returns a task id immediately and delivers its result later as its own turn, which is what makes a twenty-minute build affordable. See [Background tasks](../usage/background.md).

```text
shell_execute({"command": "cargo test --all", "background": true})
```

Like `scratchpad`, `background` is meka's own: it is consumed by the agent loop and removed from the arguments before the tool, or a remote MCP server, ever sees it.

A tool that advertises `background` itself keeps it. meka does not splice its own parameter over a name a tool already uses, and does not strip or interpret one either, so a server with a `background` color or a detach flag of its own receives the argument untouched and the call does not detach.

These two are also the only parameters meka type-checks. A `background` that is not a boolean, or a `scratchpad` that is not a string, refuses the call and says what was expected, rather than being read as absent. Both decide what a call *does* rather than what it is called with, so ignoring a wrong type would silently turn a detached call into a blocking one, or drop output the agent asked to keep. A tool's own arguments are the tool's to validate: meka reports a mismatch as an advisory on the result and lets the call through, since a remote server is the authority on what it accepts. `null` counts as absent for both, which is what models emit for an optional argument they are not using.

## Scratchpad parameter

A `scratchpad` string parameter saves a tool's output to the scratchpad under that name instead of returning it inline, so a large result stays out of the conversation.

```text
shell_execute({"command": "pdftotext doc.pdf -", "scratchpad": "pdf_text"})
```

It is honored on **every** tool, MCP servers included: the redirect happens where the result is
recorded, not inside the tool. Eleven built-ins also *advertise* it in their schema, which is how the
model discovers it: `file_read`, `file_edit`, `file_write`, `file_find`, `file_search`,
`web_fetch`, `shell_execute`, `conversation_read`, `agent_spawn`, `agent_followup`
and the `todo_*` tools, the last for uniformity alone, since its list is kept as state and nothing is redirected.

Three of those lift a cap when it is set, producing their full untruncated output: `file_find` (500
results), `file_search` (100 matches) and `web_fetch` (`limit`). An explicit `limit` on
`file_find` or `file_search` still applies.

## How tool calls work

1. The agent receives your instruction and decides which tools to call
2. For each tool call, meka checks the current permission level
3. A call above the level is refused, or put to you for approval when approvals are on
4. If permitted, the tool executes and its output is fed back to the agent
5. The agent may make additional tool calls or respond with text
6. This loop continues until the agent has no more tool calls to make

Tool calls and their results are displayed in the terminal so you can see what the agent is doing.

## `todo_write`, `todo_edit`, `todo_read`

A structured task list for a session. The agent uses it to track multi-step work and communicate progress; the list is displayed in the terminal (for the root agent) and injected into the conversation context each turn. Every call returns the full current list with task numbers, so the agent always has the numbers its next edit needs.

- `todo_write` creates or replaces the whole list: `title`, a short heading for the overall goal, and `items`, each a task string (status defaults to `pending`) or an object `{text, status}`. Tasks are numbered `1..N` in order. Both parameters are required.
- `todo_edit` updates statuses by task number: `set`, e.g. `{"1": "completed", "2": "in_progress"}`. This is the common path while working. Every number is checked before any status changes, so one bad number changes nothing.
- `todo_read` returns the list and takes no arguments.

Task statuses are `pending`, `in_progress`, `completed`, and `canceled`.

## `agent_spawn`

Spawns a sub-agent to perform research, analysis, or any other delegated task. The sub-agent gets its own private todo list (the `todo_*` tools operate on the sub-agent's own state), runs silently (its tool calls are not surfaced to the terminal), and returns a single text report. Use this to keep exploratory or speculative work out of the main conversation context.

Multiple `agent_spawn` calls in one assistant turn run in parallel; useful when independent investigations can proceed concurrently.

**Recursion.** Sub-agents may themselves spawn further sub-agents, so an agent can orchestrate a team. Nesting is bounded by [`session.subagent_max_depth`](../configuration/config-file.md#sessionsubagent_max_depth) (default 3; `1` reproduces the old "sub-agents can't spawn" behavior, `0` disables `agent_spawn` entirely). Pass the optional `max_depth` parameter to tune how deep a given subtree may recurse; a built-in absolute cap always bounds real nesting so recursion can't run away.

**Permission.** By default a sub-agent inherits the parent's permission level. Pass the optional `permission` parameter (`none` / `read` / `workspace` / `unrestricted`) to run it at a *more restricted* level: the value is clamped to the parent's level as a ceiling, so a sub-agent can never be escalated above its parent. This lets an orchestrator hand untrusted or risky work to a read-only sub-agent. A sub-agent shares its parent's approvals switch, and its prompts reach the parent's frontend.

**Writable roots.** Pass `writable_roots`, a list of directories, to confine the sub-agent's writes to exactly those: the first becomes its working directory, so relative paths in its tool calls resolve there, and the rest become its additional workspace roots. Nothing of your own workspace comes with it. Each entry must be an existing directory; a relative one resolves against your working directory. You must be at `workspace` or `unrestricted`, and at `workspace` every entry must lie inside your own [workspace boundary](../usage/permissions.md#the-workspace-boundary), so a sub-agent's reach never exceeds yours; at `unrestricted` any directory may be named. The sub-agent runs at `workspace` unless `permission` asks for less. `permission: "unrestricted"` alongside `writable_roots` is refused, since the list would then bound nothing, and so is an empty list. The bounds are recorded with the sub-agent and hold across `agent_followup`, which checks them against your reach at that moment: a session that has since dropped below `workspace`, or moved to a directory that no longer contains them, cannot resume the sub-agent.

**Tools.** Pass `deny_servers` to withhold whole MCP servers from the sub-agent (its tools, its resources, and its prompts) or `deny_tools` to withhold individual tools by name. Both union with whatever [`[subagents]`](../configuration/config-file.md#subagents) already denies; there is no way to grant something back, so a nested `agent_spawn` can only ever narrow further. Config is the place to put a restriction you always want, since the failure mode this guards against is an orchestrator forgetting to ask for it.

**Profile.** With [`[subagents].agent_chosen_profile`](../configuration/config-file.md#subagents) on, pass `profile` to run the sub-agent on another configured profile; the parameter lists every configured name. A sub-agent given a profile keeps it on every `agent_followup`, whatever profile the parent has since switched to. Only an act on the sub-agent's own session, such as an import onto another profile, moves it. Without the parameter the sub-agent runs on the parent's profile and follows it across a switch. The parent's own profile needs no naming.

**Context is granted, not inherited.** A sub-agent starts with a clean slate and receives only what you ask for:

- `memory: "read"` grants read access to your memory store. Default `"none"`, because memories from unrelated work are context the sub-agent pays for and reasons from. Sub-agents can never write to the store: record anything worth keeping yourself, from the sub-agent's report.
- `instructions: "inherit"` hands over your [instructions file](../usage/instructions.md) verbatim. Default `"none"`, because those instructions describe *you*: your persona, how to address the user, what to volunteer. A sub-agent handed one task by one of your turns is not you. Grant them when the task needs the project's standing rules and quoting the relevant ones into `prompt` would be lossy or expensive; pass a `skill` when the direction is reusable.

Neither can be granted beyond what you hold yourself, so authority only narrows going down a chain of sub-agents. A sub-agent you gave no memory cannot give its own sub-agents any.

**What it sees.** A sub-agent gets the same per-turn context the root does, over its own registry: the tool discovery index, the skill index, the memories it was granted, and the instructions of the MCP servers it is allowed to use. Its conversation, todo list and scratchpad entries are private; filesystem changes are shared within the workspace it was given.

**Follow-up.** `agent_spawn` returns the sub-agent's id on the first line of its result, above the report. Keep it if you might have a second question: with it you can call `agent_followup` instead of re-spawning one that would have to rediscover everything.

## `agent_list` / `agent_followup` / `agent_steer` / `agent_delete`

A sub-agent is not a one-shot. Its conversation persists under its own session, so you can go back to it.

- **`agent_list`**: the sub-agents this session spawned, one per line as `<id>\t<cwd>\tturns=<n>\tlast_active=<timestamp>`. Direct children only: a sub-agent's own sub-agents belong to it and appear in *its* list.
- **`agent_followup({id, prompt, scratchpad?})`**: asks a sub-agent another question. It still has its own conversation, so it can build on what it already found rather than starting from your summary of it. Returns its new report. A sub-agent that is still running, because it was spawned or followed up with `background: true`, refuses a follow-up and says so: reach it with `agent_steer`, or follow up once it has finished.
- **`agent_steer({id, message, interrupt?})`**: puts a message in the sub-agent's inbox and returns at once, with no answer. A sub-agent that is running reads it at its next round boundary, after that round's tool results; one that has finished reads it at the start of the next `agent_followup`, after the follow-up's own words. Either way it arrives under a header naming the parent as the sender and when it was sent. For a correction or a change of course while the work is under way; use `agent_followup` when you want a reply. With `interrupt: true` the sub-agent does not finish its current step first: the answer it is writing is cut and kept as far as it got, and the message is the next thing it reads, inside the same turn; a tool it is running still finishes, and the message follows that tool's result. The cost is the request sent again, so set it when the answer under way is being wasted, not merely when you have something to add. A profile that does not stream has no partial answer to keep: the reply being generated is dropped whole and the request goes again with the message.
- **`agent_delete({id})`**: discards a sub-agent: its conversation, its scratchpad entries, and any sub-agents it spawned in turn. Nothing it wrote to disk is touched. Worth doing once you have what you needed, so a long session isn't carrying every sub-agent it ever ran.

`agent_followup`, `agent_steer` and `agent_delete` take the full id, or any prefix that is unique among this session's sub-agents; a prefix two of them share is refused and both are named. All three refuse an id that isn't a child of the current session, so one session can never drive, steer or delete another's sub-agents.

**All five go together.** Denying `agent_spawn` in [`[tools].disabled_tools`](../configuration/config-file.md#tools-built-in-tool-filters), or setting [`session.subagent_max_depth = 0`](../configuration/config-file.md#sessionsubagent_max_depth), removes the four lifecycle tools too: an agent that cannot delegate has no sub-agents for them to act on, and leaving them behind would let it drive the ones a previous run left in the store. `meka tool list` reports all five as `disabled` in either case. Denying only `agent_list` removes just that one.

**A follow-up runs under the terms of the spawn, not your current ones.** The permission level, the deny lists, the memory level and the inherited scratchpad names are recorded when the sub-agent is created and replayed on every follow-up. If you spawned a sub-agent at `read` and have since switched to `unrestricted`, following up on it still runs it at `read`. That is deliberate: otherwise a second question would be a way to escalate a sub-agent you deliberately restricted. A sub-agent that shares your workspace keeps the working directory it was spawned in; at `workspace`, a follow-up is refused once that directory lies outside your own boundary, the same check a sub-agent's `writable_roots` get.

Two things do *not* survive a follow-up, because they only ever lived in memory: the sub-agent's todo list, and which files it had read. It is told as much at the start of the turn. Its context gauge does survive: the follow-up starts from the occupancy the sub-agent's row last recorded, so its first turn back is checked against the ceiling like any other.

One follow-up at a time per sub-agent. A second concurrent call on the same sub-agent is refused rather than interleaved, since both would be appending to one conversation from a view of it that the other has already changed.

## The `skill_*` tools

Skills are knowledge packages stored in `~/.config/meka/skills/<name>/SKILL.md`. The per-turn context lists the installed ones with their descriptions; these tools open, search, and (when enabled) maintain them.

- `skill_read({"name": "<skill-name>"})` returns the full body, prefixed with the skill's base directory.
- `skill_search({"pattern": "<regex>"})` matches each line of every skill, bodies included. This is what reaches skills the capped index did not list, and what answers "which of my skills covers this" when the one-line descriptions do not.
- `skill_write({"name": ..., "description": ..., "priority": ..., "body": ...})` creates or updates a skill. Omitting `body` keeps the existing one.
- `skill_delete({"name": ...})` removes the skill's whole directory, bundled files included.

The last two are registered only when [`[skills] agent_managed`](../configuration/config-file.md#skills) is on, and never for a sub-agent. See [Skills](../usage/skills.md) for how to author skills and [Letting the agent manage skills](../usage/skills.md#letting-the-agent-manage-skills) for when to hand authoring to the agent.

## `image_render`

Displays an image the agent has in memory, as base64 bytes or in a scratchpad entry, as a multimodal content block. Complements `web_fetch` (network) and `file_read` (local file) by covering the third case: image data produced on the fly by a command pipeline.

Typical workflow:

```text
shell_execute({"command": "ffmpeg -i input.mp4 -vframes 1 -f image2pipe pipe: | base64 -w0", "scratchpad": "frame"})
image_render({"from_scratchpad": "frame"})
```

Parameters:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `from_scratchpad` | string | one of two | Name of a scratchpad entry containing base64-encoded image bytes |
| `base64` | string | one of two | Base64-encoded image bytes, passed inline |

Exactly one of `from_scratchpad` or `base64` must be provided. Prefer `from_scratchpad` for large images; inline base64 inflates tool-call JSON.

The bytes must decode to a supported raster image. PNG, JPEG, GIF, WebP, and BMP pass through unchanged; TIFF, ICO, HDR, EXR, TGA, PNM, QOI, DDS, and Farbfeld are auto-converted to PNG. Size cap is ~3.75 MB on the final payload.

Only call `image_render` when the current model supports vision input.

## `conversation_search` / `conversation_read`

Search and re-read this session's **full** conversation, including earlier turns that [compaction](../usage/interactive-mode.md#compact) summarized and removed from the model's context. Compaction never deletes turns (it appends a boundary and hides the older ones); these tools read straight from the on-disk event log, so a detail the compaction summary dropped is still recoverable.

`conversation_search` searches and returns matching lines, each tagged with a message index (`#N`) and role:

```text
conversation_search({"query": "auth token", "is_regex": false, "limit": 20})
```

- `query` (required): text to search for; a literal substring (case-insensitive) unless `is_regex` is set.
- `is_regex`: treat `query` as a case-sensitive regular expression. Default: `false`.
- `limit`: maximum matches to return (capped at 100). Default: 20.

`conversation_read` reads turns by the `#N` index that `conversation_search` reports:

```text
conversation_read({"start": 47, "limit": 3})
```

- `start` (required): 1-based message index to read from.
- `limit`: number of consecutive messages to read (max 20). Default: 1.
- `scratchpad`: save the output to a scratchpad entry instead of returning it inline.

After a compaction, the summary message reminds the agent that these tools exist. Large tool outputs appear as `<large-output>` references in both `conversation_search` and `conversation_read` results (rather than inlining the full payload); read their full content with `scratchpad_read`.

## `context_check` / `context_compact`

Where `conversation_*` reads the **archive** (the full log on disk, including turns compaction removed from the window entirely), `context_*` manages the **live window**.

`context_check` takes no arguments and reports the current state:

```text
Using 84000 of 200000 tokens (42%).
Headroom: 86000 tokens before the context ceiling at 85%. Auto-compaction fires there,
between turns or between two of your tool rounds.
Compaction may keep about 16000 tokens of recent rounds verbatim; the rest
becomes a summary.
Fixed overhead: about 12000 tokens of system prompt and tool schemas (estimated).
Compaction does not reclaim this.
Conversation: about 72000 tokens, which is the part compaction acts on.
Compactions so far: none, so nothing has been summarized away yet.
```

This exists because the pushed `[Context budget]` block is rendered once, at the start of a turn, and so does not move while the agent works. During a long tool loop it is stale. See [What the agent sees](../usage/sessions.md#what-the-agent-sees). The headroom is net of what this round's whole `scratchpad_read` calls have already taken, so it is the room the next read gets.

`context_compact` requests a compaction before the agent's next step. It runs once the current batch of tool calls finishes, and the turn then continues against the summary; one request is honored per turn.

- `instructions`: what to preserve or drop. Active restrictions, authorizations, and commitments must still be preserved.
- `keep_recent`: allow recent rounds to be kept verbatim. Default `true`; a tail is kept only when enough older history can be summarized. `false` requests no retained tail.

There is a third tool, `context_replace`, that exists only inside a checkpoint turn and is how the agent submits its summary. It is deliberately absent from the ordinary catalog and from `[tools]` configuration. See [Compacting a session](../usage/sessions.md#compacting-a-session).
