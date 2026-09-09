# CLI options

```text
meka [OPTIONS] [COMMAND]
```

## Commands

### `account`

Manage accounts and their credentials. `meka account add` writes an `[accounts.<name>]` table to
`~/.config/meka/config.toml` and keeps its secret in the store; `usage`, `whoami` and `stats`
are the read-only views described under [Account info](../usage/account.md).

```bash
meka account add anthropic --backend claude-subscription
meka account list [--format <FORMAT>]
meka account login anthropic
meka account remove anthropic
meka account rename anthropic claude
meka account usage [--profile <NAME>] [--format <FORMAT>]    # session / weekly windows
meka account whoami [--profile <NAME>] [--format <FORMAT>]   # plan, tier, org, role and local auth status
meka account stats [--profile <NAME>] [--format <FORMAT>]    # lifetime tokens, streaks, per-day counts
```

See the [`meka account` CLI reference](./config-file.md#meka-account-cli) for the full flag list.

### `profile`

Manage profiles: an account plus a model and every model-tied setting. `meka profile add` writes a
`[profiles.<name>]` table; nothing here touches a credential.

```bash
meka profile add work --account anthropic --model claude-opus-5
meka profile list [--format <FORMAT>]
meka profile set work model claude-opus-5
meka profile set work effort --unset   # back to the default
meka profile use work
meka profile remove work
meka profile rename work daily
```

See the [`meka profile` CLI reference](./config-file.md#meka-profile-cli) for the full flag list.

### `session`

Manage stored sessions: list them, show one in full, export one as Markdown or JSON, import a JSON
export, fork or rewind one, or delete them.

Every `<SESSION_ID>` is a full id or any unique prefix of one, which is what the listings print.

```bash
meka session list [-n <LIMIT>] [--include-children] [--format <FORMAT>]   # default limit: 20; sub-agent sessions hidden unless asked
meka session show <SESSION_ID> [--format <FORMAT>]    # full id, cwd, permission, title
meka session export <SESSION_ID> [-o <OUTPUT>] [--format <FORMAT>]   # markdown (default) or json; -o - prints to stdout
meka session import <INPUT>                           # a JSON export; - reads stdin
meka session fork <SESSION_ID>                        # prints the copy's id
meka session rewind <SESSION_ID> [-n <TURNS>]         # default 1
meka session delete <SESSION_IDS>...
meka session delete --older-than-days <DAYS>
meka session delete --all
```

See [Sessions](../usage/sessions.md#exporting-a-session) for details.

### `history`

View or clear the REPL input history that powers Up-arrow / Ctrl+R recall (distinct from saved
sessions and from the `/history` slash command).

```bash
meka history list [-n <LIMIT>] [--format <FORMAT>]   # default 50; -n 0 shows all
meka history clear
```

### `mcp`

Manage MCP servers in `config.toml` and their stored credentials.

```bash
meka mcp list [--format <FORMAT>]
meka mcp get <NAME> [--format <FORMAT>]
meka mcp add <NAME> [LOCATION] [ARGS]... [flags]   # a URL (HTTP) or an executable (stdio)
meka mcp remove <NAME>
meka mcp disable <NAME>
meka mcp enable <NAME>
meka mcp reconnect <NAME>
meka mcp tools <NAME> [--format <FORMAT>]
meka mcp login <NAME> [--auth-token-stdin | --client-secret-stdin]
meka mcp logout <NAME>
```

See [MCP](../usage/mcp.md#meka-mcp-cli) for the `add` flags and what each command does.

### `tools`

Inspect the built-in tool filters. `meka tools list` prints every built-in with its `Required`
permission, the `Source` of that requirement (`builtin`, or `override` from `[tools]`), its
`Visibility` (`enabled`, `deferred`, or `disabled`), and the first line of its description.

```bash
meka tools list [--format <FORMAT>]
```

See [Filtering built-in tools](../tools/overview.md#filtering-built-in-tools).

### `skill`

Manage the skills under `~/.config/meka/skills/`.

```bash
meka skill list [--paths] [--format <FORMAT>]
meka skill get <NAME> [--format <FORMAT>]    # frontmatter and on-disk paths
meka skill show <NAME> [--format <FORMAT>]   # the rendered body
meka skill add <NAME> [flags]
meka skill remove <NAME>
```

See [Skills](../usage/skills.md#meka-skill-cli) for the `add` flags.

### `memory`

Manage the agent's saved memories.

```bash
meka memory list [--format <FORMAT>]
meka memory get <NAME> [--format <FORMAT>]    # every stored field
meka memory show <NAME> [--format <FORMAT>]   # the body
meka memory add <NAME> --description <DESCRIPTION> [flags]
meka memory edit <NAME>      # the body, in $VISUAL, then $EDITOR
meka memory remove <NAME>
meka memory verify [--rebuild]
meka memory export [--dir <PATH>]   # default: ./meka-memory-export
```

See [Memory](../usage/memory.md#cli) for the `add` flags.

### `instructions`

Show the standing instructions the agent is given.

```bash
meka instructions show   # the resolved text and where it came from
meka instructions path   # the paths checked, and whether each exists
```

See [Instructions](../usage/instructions.md).

### `schedule`

Inspect and cancel the wakeups the agent scheduled for itself. There is no `create`: a job needs a
session for its turn to run in, and the agent creates one through `schedule_create`.

Every `<ID>` is a full id or any unique prefix of one, which is what `list` prints.

```bash
meka schedule list [--session <SESSION>] [--format <FORMAT>]   # every session's jobs, or one session's by id or prefix
meka schedule show <ID> [--format <FORMAT>]   # full prompt, gate command, session, withheld reason
meka schedule cancel <ID>
```

See [Scheduling](../usage/scheduling.md) for details.

### `acp`

Run meka as an [ACP](../usage/acp.md) agent over stdio. Takes no flags of its own; `-c` and `-r`
are refused, and `--profile` selects the profile a new session starts on.

```bash
meka acp
```

### `serve`

Run meka as a long-lived [HTTP service](../usage/http-api.md). `--bind <ADDR>` overrides
`[serve].bind`. Like `acp`, it refuses `-c` and `-r`: the host creates a session per request.

```bash
meka serve
meka serve --bind 0.0.0.0:8080
```

## Options

### `-p`, `--prompt <TEXT>`

Run the agent's first turn immediately with this text as the user message, then drop into the REPL for follow-up. Pair with [`--oneshot`](#--oneshot) to exit after the first turn instead of opening the REPL. `-p -` reads the prompt from stdin, to end of input.

```bash
meka -p "list all files larger than 1MB in the current directory"   # first turn, then REPL
meka --oneshot -p "list all files larger than 1MB"                  # first turn, then exit
git diff | meka --oneshot -p -                                      # the prompt is the diff
```

When omitted, meka starts the REPL with no initial input. There is no bare positional prompt, so a mistyped subcommand is an error rather than a session opened with the typo as its first turn.

### `-c`, `--continue`

Continue the most recent session. Takes no value.

```bash
meka -c                        # pick up where you left off
meka -c -p "and now add tests" # …with an opening prompt
```

Starting fresh when there is no session yet is not an error; meka just begins a new one.

### `-r`, `--resume <SESSION>`

Resume a specific session. Accepts either the full id or any unique leading prefix.

```bash
meka -r 550e8400-e29b-41d4-a716-446655440000     # full id
meka -r 550e                                     # prefix; works if unique
meka -r 550e -p "and now add tests"              # …with an opening prompt
```

Errors if the session does not exist, the prefix matches multiple sessions (with the matching ids listed for disambiguation), or the session is locked by another meka instance.

`-c` and `-r` are mutually exclusive. Both work with `--oneshot`, which runs a single turn against the session and exits:

```bash
meka --oneshot -r 550e -p "summarize what we decided"
```

### `--permission <LEVEL>`

Set the initial permission level. Accepts `none`, `read`, `workspace` or `unrestricted`.

```bash
meka --permission workspace
```

There is no flag for [approvals](../usage/permissions.md#approvals): a new session takes `[permissions].approvals` from the config file, and `/approvals on` moves it afterwards.

Default: `read`.

Recorded on the session, so a resume comes back at the level the session was last at rather than at
the default. Passing `--permission` alongside `-c` / `-r` repins it, the way `--profile` does. A
level that is no longer in `[permissions].enabled` is not granted on resume: the session drops to
the configured default with a warning.

### `--writable-root <PATH>`

Add a directory to the workspace, so writes may land there at `workspace` permission. Repeatable.
The working directory is always a root; this adds to it.

```bash
meka --permission workspace --writable-root ../shared-assets --writable-root /srv/build
```

Deliberately a flag rather than a config key: which folders this run may write is a per-run scope,
like the working directory itself, not a preference to persist.

A path that does not resolve at startup is reported as a warning and kept, so a build directory that
does not exist yet becomes a root the moment it does. A path that is not a directory, or a system
directory the sandbox masks, is refused with a warning: neither can be expressed as a boundary by
every backend.

The masked set is the filesystem root itself plus `/proc`, `/dev`, `/sys`, `/run`, `/tmp` and
`/var/tmp`, and `/run/user` and `$XDG_RUNTIME_DIR` as whole subtrees. A root is refused when it *is*
one of these and when it is an **ancestor** of one, but not when it is merely underneath: a root
under one of these is usually fine and refusing it would be a real loss, since
`/run/media/$USER/drive` is an ordinary external disk. The ancestor half is why `--writable-root
/var` is refused, and it is also why `--writable-root ~` is refused on WSL and on minimal window
managers, where `$XDG_RUNTIME_DIR` lives under `$HOME` and so binding `$HOME` would hand the session
bus back. `/tmp` and `/var/tmp` are in the set because
Bubblewrap masks them with a tmpfs and then binds the requested root back over it, last mount
winning: binding `/tmp/work` restores just that directory, but binding `/tmp` restores the entire
host `/tmp` including every X11, D-Bus and tmux socket in it, which is a route straight back out of
the sandbox. The cost is that a session started with `cd /tmp` has no write boundary, which is the
safe direction to fail.

The flag reaches the REPL, one-shot runs, and ACP sessions. It does **not** reach sessions created
through `POST /v1/sessions`, which are single-root by design, and therefore does not reach a
scheduled turn under `meka serve` either: those run in the session the job belongs to, which the
HTTP API created.

### `--sandbox-backend <BACKEND>`

Pick the Linux sandbox backend for this run, `landlock` or `bubblewrap`. Wins over
`MEKA_SANDBOX_BACKEND` and [`[shell].sandbox_backend`](./config-file.md#shellsandbox_backend), and
like either of those, pinning a value suppresses the install-Bubblewrap warning the auto-pick
prints. Ignored on macOS and Windows.

```bash
meka --sandbox-backend landlock
```

### `--profile <NAME>`

Select which configured profile a session runs on. Takes the name of a profile from
`[profiles.<name>]`, overriding `default_profile` in the config file. The choice outlives the run:
on a new session it is what gets recorded, and on a resume it rewrites the row.

```bash
meka --profile work
```

The value is a profile name (e.g. `work`, `personal`), not an account or a backend. List configured profiles with `meka profile list`. There is no short form: `-p` is the prompt.

A new session records the profile it runs on, so `meka -c` later comes back on it rather than on
`default_profile`. Passing `--profile` alongside `-c` / `-r` **repins** the session: the row is
rewritten and it keeps that profile from then on. See
[what a resume restores](../usage/sessions.md#what-a-resume-restores).

> **There is no `--model`, `--base-url`, `--thinking` or `--thinking-budget`.** A profile is an
> indivisible bundle of an account, a model and every model-tied setting, and a session records
> which one it runs on rather than a rewritten copy. To change a setting, edit the profile with
> [`meka profile set`](./config-file.md#meka-profile-cli); to run something different, make a second
> profile and select it with `--profile`.

### `--no-stream`

Disable streaming for this run. The agent waits for the complete response before displaying it. By default, responses are streamed token-by-token; [`display.stream`](./config-file.md#displaystream) is the persistent form.

```bash
meka --no-stream
```

### `--render-mode <RENDERER>`

Markdown render mode. Accepts `termimad` (default), `syntect`, or `raw`.

- `syntect`: Syntax-highlighted markdown source, including per-language code blocks. Nothing is reflowed, so a table with long cells runs past the terminal width.
- `termimad`: Rendered markdown, reflowed to the terminal: paragraphs re-wrap, wide tables wrap inside their box, and markers are consumed rather than shown. meka parses the CommonMark itself, so `-`/`+` bullets, `__bold__`, `_italic_`, ordered lists, and links all render. Colors come from the same theme as `syntect`, and fenced code blocks are syntax-highlighted by it.
- `raw`: Raw markdown printed verbatim with aligned tables.

`termimad` is the default: meka's own output is table-heavy (`task_list`, `scratchpad_list`, anything the model tabulates), and those run past the right edge under `syntect`. Pick `syntect` when you want to see the markdown source as the model wrote it.

```bash
meka --render-mode raw
```

Can also be set permanently via `display.render_mode` in the config file.

### `--instructions <STRING>`

Standing [instructions](../usage/instructions.md) for this run, replacing the `instructions.md` file and both `MEKA_INSTRUCTIONS*` environment variables. Takes the text itself, not a path; use `"$(cat file.md)"` to read one.

```bash
meka --instructions "Be terse. No code fences in answers."
```

### `--skill <NAME>`

Invoke a [skill](../usage/skills.md) as the first turn. Mirrors the REPL slash command [`/skill <name> [extra...]`](../usage/skills.md#invoking-a-skill-from-the-cli). `-p`, if given, is prepended to the rendered skill body as additional context. Pair with [`--oneshot`](#--oneshot) to exit after the turn instead of opening the REPL.

```bash
meka --skill download-videos -p "https://example.com/video"             # first turn, then REPL
meka --skill download-videos --oneshot -p "https://example.com/video"   # first turn, then exit
```

Errors out with a clean message if the skill name is unknown.

### `--oneshot`

Exit after the first turn finishes. Requires either `-p` or `--skill <NAME>`; without one of those, meka has nothing to do. Useful for scripts and CI invocations.

```bash
meka --oneshot -p "summarize the last commit"
meka --oneshot --skill deploy -p "to staging"
```

### `--format <FORMAT>`

What a `--oneshot` run writes to stdout: `plain` (the default) streams the answer as text; `json` prints nothing during the turn and one object when it ends. See [One-shot mode](../usage/one-shot-mode.md#json-output) for the object's fields. The flag applies to `--oneshot` alone; a run without it is refused. `meka account usage`, `whoami` and `stats` take the same flag and values, and `meka session export --format` takes `markdown` or `json`.

```bash
meka --oneshot -p "what changed?" --format json | jq -r .text
```

Every listing and `show` command takes the same `--format plain|json`: `session list|show`,
`account list`, `profile list`, `mcp list|get|tools`, `schedule list|show`, `memory list|get|show`,
`tools list`, `history list` and `skill list|get|show`. Under `json`, a `show` prints one object and
a listing prints `{"<nouns>": [...]}` (`sessions`, `accounts`, `profiles`, `servers`, `jobs`,
`memories`, `tools`, `history`, `skills`), with the field names the [HTTP API](../usage/http-api.md)
uses for the same object where it has one. An empty listing is the envelope around an empty array
and nothing on stderr; in `plain`, it is a `No <nouns>.` note on stderr and nothing on stdout.
Warnings and hints stay on stderr under either format, so `meka … --format json 2>/dev/null | jq`
sees only the document.

```bash
meka session list --format json | jq -r '.sessions[].id'
meka mcp get notion --format json | jq .url
```

### `--eager-load-tool <SERVER:TOOL>`

Eager-load a specific MCP tool for this session, bypassing the `load_tool` round-trip. The tool's schema ships in the cacheable tools-array prefix from turn 1 instead of being deferred. Mirrors the per-server [`eager_load_tools`](./config-file.md#mcpservers) config field: repeatable, raw tool names (the server-advertised form, not `mcp__<server>__<tool>`).

Particularly useful for scripted runs that know up front which tools they'll need. The flag *appends to* whatever `eager_load_tools` lists in `config.toml` for that server; it doesn't replace existing entries. Unknown server names log a warning and are skipped.

```bash
meka --eager-load-tool notion:search --eager-load-tool github:create_issue \
     --oneshot -p "search Notion for the deploy runbook and open a GitHub issue"
```

### `-v`, `--verbose`

Increase log verbosity. Can be repeated up to three times.

```bash
meka -v      # info
meka -vv     # debug
meka -vvv    # trace
```

### `-h`, `--help`

Print help. `-h` is the summary; `--help` adds the longer prose under the flags that have it.

### `-V`, `--version`

Print version.
