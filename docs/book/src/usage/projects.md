# Projects

A project is a named work context a session opens. It is defined in `config.toml`, a session is put on one only by an explicit act, and what it gives the session is a [memory tier](./memory.md#tiers) of its own beside the global one: the notes of that project, seen by every session on it and by no session off it.

A project is an idea, not a directory. Nothing binds one to a path, nothing opens one because of where meka was started, and a session with nothing named is on no project. Everything the agent writes about a project while on it is filed under the project; everything it writes while on none is global.

## Defining a project

A project is a `[projects.<name>]` table in `config.toml`, after `[mcp]`. An empty table is a complete project:

```toml
[projects.webdev]

[projects.movie]
```

`meka project add <name>` writes the table and places it there. Names follow the rule memories and skills follow: letters, digits, `-` and `_`, starting with a letter or digit, and never a name Windows reserves. Names are compared exactly, and two that differ only by case are refused, in the file and at `project add`.

```bash
meka project add webdev
meka project add movie --allowed-server blender --disabled-skill release-checklist
meka project list [--format <FORMAT>]     # name, memories, sessions, servers, skills, configured
meka project set movie allowed_skills storyboard,shotlist   # replaces the list
meka project set movie allowed_skills --unset               # back to every installed skill
meka project remove movie                 # refused while any session or memory is on it
meka project rename webdev web            # moves every session and memory with it
```

`project list` shows every project the file defines and every one a row still names, with how many memories and root sessions each holds and the view it gives them (`all`, the names allowed, or `all but` the names disabled); a name only the store carries, because it left the file by hand while rows still named it, is listed with `configured` `no`. `rename` moves the rows and the table in one step, and `remove` refuses with the counts until nothing is on the project: `project 'web' has 3 memories and 2 sessions; move or remove them first`. `add` and `set` refuse a server no `[[mcp.servers]]` entry defines and a name in both lists of a pair, before anything is written.

## Opening a session on a project

Every surface selects a project by name, and a session created from another session copies its project: a sub-agent spawned by `agent_spawn`, a fork, an imported archive. A sub-agent keeps following its parent: each `agent_followup` runs it in the project the parent is on then and records that on its row, as it does the parent's profile; the narrowing its spawn recorded stays.

On the command line, `--project <name>` opens a new session on the project, and beside `-c` or `-r` it follows one rule: it **filters a selection meka makes and moves a session you named**.

```bash
meka --project webdev                       # a new session on webdev
meka -c --project webdev                    # the latest session on webdev, or a new one there
meka -r --project webdev                    # the picker over webdev's sessions
meka -r 550e --project webdev               # moves session 550e onto webdev
```

On the screen, `/project` opens a picker over the configured projects, with a first row that leaves the project; `/project open <name>` moves the session directly and `/project close` takes it off its project. Either way the row is rewritten, so the choice holds for a resume from any surface, and the next turn's context restates the memories the session now sees. `/status` names the project the session is on.

Over HTTP, `POST /v1/sessions` takes `project`, `PATCH /v1/sessions/{id}` moves a session with `{"project": "movie"}` and takes it off one with `{"project": ""}`, every session record carries `project`, and `GET /v1/sessions?project=<name>` lists the sessions on one (`project=` those on none). `GET /v1/projects` lists the projects. See [HTTP API](./http-api.md#sessions).

Under ACP, a `project` entry in `configOptions` is offered whenever the file defines a project, with a "No project" value first; see [ACP](./acp.md#session-config-options).

A session whose project has since left `config.toml` is refused when it next runs, with the configured names, as a session on a deleted profile is: add the project back, or move the session with `meka -r <id> --project <name>`.

## What a project changes

Two things: the memory tier, and what the session sees of the installed servers and skills.

A session on a project reads the global memories and the project's, `memory_write` files into the project unless the model asks for `tier: "global"`, and the `[Memory]` section the model reads is grouped by tier; the rules are on the [Memory](./memory.md#tiers) page.

A project may narrow the view with four lists, documented under [`[projects.*]`](../configuration/config-file.md#projects): `allowed_servers` and `disabled_servers` for MCP servers, `allowed_skills` and `disabled_skills` for skills. The default view is everything installed; an `allowed_` list names the whole view and a `disabled_` list subtracts from it. The view is read live, so `/project open` and a `PATCH` change what the next turn sees, and the per-turn context announces the tools that arrived or left as it does for any catalog change. A sub-agent spawned from the session is narrowed the same way, on top of what its spawn call and `[subagents]` already withhold. A scheduled job's gate is checked against the view when the job is created, by the session or over HTTP; a job created before a move keeps its probe, as a sub-agent keeps the scope it was spawned with.

The profile, the permission level, the working directory and the standing instructions stay the session's own, as they are on any session.
