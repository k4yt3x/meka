# Projects

A project is a name a session is on. It groups the session with its sibling sessions and gives them a [memory tier](./memory.md#tiers) of their own beside the global one: the notes of that project, seen by every session on it and by no session off it.

A project is an idea, not a directory. Nothing binds one to a path, nothing opens one because of where meka was started, and a session with nothing named is on no project. Everything the agent writes about a project while on it is filed under the project; everything it writes while on none is global.

## Defining a project

A project is a row in the store, created by `meka project add` or `POST /v1/projects`. Names follow the rule memories and skills follow: letters, digits, `-` and `_`, starting with a letter or digit, and never a name Windows reserves. Names are compared exactly, and two that differ only by case are refused.

```bash
meka project add webdev
meka project list [--format <FORMAT>]     # name, when it was added, memories, sessions
meka project rename webdev web            # moves every session and memory with it
meka project remove movie                 # refused while any session or memory is on it
```

`project list` shows every project with when it was added and how many memories and root sessions it holds. `rename` moves the row and every session and memory on it in one transaction, and `remove` refuses with the counts until nothing is on the project: `project 'web' has 3 memories and 2 sessions; move or remove them first`. A project lives in the store, so a running `meka serve` offers one the moment it is added.

## Opening a session on a project

Every surface selects a project by name, and a session created from another session copies its project: a sub-agent spawned by `agent_spawn`, a fork, an imported archive. A sub-agent keeps following its parent: each `agent_followup` runs it in the project the parent is on then and records that on its row, as it does the parent's profile. A name no project has is refused at every one of these doors, with the names that exist, so a typo never opens an empty tier.

On the command line, `--project <name>` opens a new session on the project, and beside `-c` or `-r` it follows one rule: it **filters a selection meka makes and moves a session you named**.

```bash
meka --project webdev                       # a new session on webdev
meka -c --project webdev                    # the latest session on webdev, or a new one there
meka -r --project webdev                    # the picker over webdev's sessions
meka -r 550e --project webdev               # moves session 550e onto webdev
```

On the screen, `/project` opens a picker over the projects, with a first row that leaves the project; `/project open <name>` moves the session directly and `/project close` takes it off its project. Either way the row is rewritten, so the choice holds for a resume from any surface, and the next turn's context restates the memories the session now sees. `/status` names the project the session is on.

Over HTTP, `POST /v1/sessions` takes `project`, `PATCH /v1/sessions/{id}` moves a session with `{"project": "movie"}` and takes it off one with `{"project": ""}`, every session record carries `project`, and `GET /v1/sessions?project=<name>` lists the sessions on one (`project=` those on none). See [HTTP API](./http-api.md#sessions).

The projects themselves are managed over HTTP as they are on the command line, under the `memory:w` scope, since a project is a memory tier: `GET /v1/projects` lists them with their counts, `POST /v1/projects` with `{"name": "webdev"}` adds one and answers `201` with the project, `PATCH /v1/projects/{name}` with `{"name": "web"}` renames one and answers it under its new name, and `DELETE /v1/projects/{name}` removes one. A name a project already has, or one the rule refuses, is a `422`; a name no project has is a `404` on the path; and the removal is a `422` whose `detail` carries the counts while anything is on the project.

Under ACP, a `project` entry in `configOptions` is offered whenever a project exists, with a "No project" value first; see [ACP](./acp.md#session-config-options).

A session whose project is gone from the store, which only an edit outside meka can bring about since `project remove` refuses while a session is on it, is refused when it next runs, with the names that exist: add the project back, or move the session with `meka -r <id> --project <name>`.

## What a project changes

The memory tier, and nothing else. A session on a project reads the global memories and the project's, `memory_write` files into the project unless the model asks for `tier: "global"`, and the `[Memory]` section the model reads is grouped by tier; the rules are on the [Memory](./memory.md#tiers) page.

The profile, the permission level, the working directory, the standing instructions, the MCP servers and the skills are the installation's and the session's own, as they are on any session.

## Taking a project elsewhere

A project's memories move with an export: `meka memory export --project <name>` writes them, and on the other installation `meka project add <name>` followed by `meka memory import <dir>` restores them, stamps and read counts included. See [Export and import](./memory.md#export-and-import).
