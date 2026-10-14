# Permissions

meka uses a four-level permission system to control what tools the agent can use, and one switch,
**approvals**, that decides what happens to a call the level does not cover. Together they give you
control over the agent's capabilities and prevent accidental modifications.

## Permission levels

| Level | Indicator | What it allows |
|-------|-----------|----------------|
| **None** | `[n]` (green) | No tools without approval. |
| **Read** | `[r]` (yellow) | Read-only tools: `file_read`, `file_find`, `file_search`, `web_fetch`, `shell_execute` (sandboxed read-only), `checklist_*`, `agent_spawn`, scratchpad tools |
| **Workspace** | `[w]` (orange) | File and shell writes stay inside workspace roots. Unconfined MCP calls may need approval or be refused; the shell needs an available sandbox |
| **Unrestricted** | `[u]` (red) | Every tool, no boundary. `shell_execute` runs with no sandbox at all |

The ladder is ordered by **reach**: each level contains the ones below it, and a tool call that
needs more than the session's level is refused. With [approvals](#approvals) on, it is put to you
instead.

## The workspace boundary

At `workspace`, a write may land under:

- the working directory (which `/cd` moves),
- any folder an ACP client supplied as an additional directory,
- any `--writable-root <PATH>` you passed, repeatable.

Roots are resolved to their canonical form, so a symlink inside the workspace that points out of it
resolves to where it actually lands and is refused. A root that does not exist is dropped rather
than trusted; if none resolve, nothing is writable. A `--writable-root` that does not resolve at
startup is reported as a warning, and kept: a build directory that does not exist yet becomes a root
the moment it does.

**The boundary follows the working directory.** It is recomputed on every write rather than fixed
when the session starts, so `/cd /etc` at `workspace` makes `/etc` writable from that point on. This
is deliberate: the working directory *is* the workspace, and a boundary that stayed behind after you
moved would refuse writes to the place you are plainly now working in. The agent has no tool that moves the working
directory, so it cannot relocate its own boundary. You can, with `/cd`; and under `meka serve` a
client holding `sessions:w` can, with `PATCH /v1/sessions/{id}`.

Because the boundary follows the directory, the directory is recorded on the session row and a
[resume reopens it](./sessions.md) rather than adopting your shell's. Resuming a `workspace` session
from `$HOME` would otherwise make your whole home directory writable without you asking.

One consequence worth knowing: a *relative* `--writable-root` resolves against your shell, not
against the session. `meka -c --writable-root build` run from `~` grants `~/build`, while the
session itself may reopen in `~/project`. That follows from the flag belonging to the process rather
than to the session; pass an absolute path when you mean a directory inside the session's.

`--writable-root` belongs to the process and reaches the screen, a one-shot run, and an ACP session.
It does **not** reach a session created through `POST /v1/sessions`: the HTTP API is single-root, so a session there is confined to its own
`cwd` and nothing else. Extra roots supplied by an ACP client apply to that client's session only,
and meka does not report your `--writable-root` back to the client as though the client had asked
for it.

Because it belongs to the process, it is not recorded on the session either, and resuming a session
does not bring it back: pass it again. This is the difference between it and the profile and
permission level, which *are* recorded and *do* come back. Writing it to the row would mean a
`meka serve` sharing the data directory could later grant those roots to a job it fires, on the
authority of a flag that process was never given.

The same set governs both halves, derived once so they cannot disagree: the `file_*` tools check it
before writing, and the shell sandbox is built from it. A refusal from `file_write` names the roots
so the agent can retry somewhere valid.

A sub-agent can be handed a narrower boundary than its parent's: [`agent_spawn`'s
`writable_roots`](../tools/overview.md#agent_spawn) names the directories it may write under, each
of which must lie inside the parent's own.

## The read boundary

Reads are not confined by default: the workspace boundary says "this cannot change things outside
the workspace", not "this cannot see them". `[permissions].confine_reads = true` adds the second
promise. Below `unrestricted`, a read, by a file tool or by a command, may then land only under:

- the session's roots, the same list writes are confined to: the working directory, an ACP client's
  additional directories, each `--writable-root`;
- the read-only roots a parent handed a sub-agent through [`agent_spawn`'s
  `readable_roots`](../tools/overview.md#agent_spawn);
- the system directories, where the operating system keeps the programs, libraries and
  configuration a command needs to run at all, and anything [`[shell].readable_paths`](../configuration/config-file.md#shellreadable_paths) adds. The home directory is not among them: that is the point;
- the skill roots [`[skills].extra_paths`](../configuration/config-file.md#skills) names, so a
  skill's bundled files stay readable, as meka's own skills store does.

The switch is instance-wide and read at startup, so every session the process runs is bound by it;
`unrestricted` remains the per-session way out, since it disclaims every boundary. The same set
governs both halves, derived once: `file_read`, `file_find`, `file_search` and
`scratchpad_load_file` refuse a path outside it and name the roots, `file_find` and `file_search`
step around what a symlink reaches outside it, and the shell sandbox is built from it. The
environment block the agent reads each turn names the readable roots and the fact that the
system directories are readable too.

What a command no longer sees, it mostly does not see at all: under Bubblewrap the root of the
sandbox is an empty filesystem with only the readable paths bound into it, so a path outside them
is absent rather than refused, and a symlink out of the roots points at nothing. Two consequences
worth knowing. Anything a tool reads from the home directory is gone, which covers `~/.cargo/bin`,
`~/.local/bin`, version-manager shims, the global git configuration and every package cache; a
directory among them comes back by being named as a root, and nothing else does. And a path's
existence, name and size may still be visible under Landlock alone and under macOS, where `stat`
is not mediated so that paths resolve through their ancestors; the bytes are what the boundary
protects.

The switch is the deployment's default, not the only way in. With it off, a parent still confines
a sub-agent's reads by naming `readable_roots`: that sub-agent, and anything it spawns, reads its
workspace, the list and the system directories, the way `writable_roots` hands a sub-agent a
write boundary from a parent that has none. A spawn turns confinement on, never off. An entry
inside the sub-agent's workspace is refused, since the workspace is readable already; the
refusal names `writable_roots`, which is how a sub-agent is confined to a directory. A
sub-agent's reach never exceeds its parent's here either: `readable_roots` are admitted against
the parent's own readable roots when the parent's reads are confined, a parent reading everywhere
may name anywhere, and a follow-up re-admits the worker's directory and roots against the parent's
reach at that moment. A sub-agent whose call names nothing holds its parent's boundary, and one
handed a list runs at `workspace` unless its call asks for less.

The file tools refuse a path outside the roots in the same words whether or not it exists, and
name it as it was given rather than where a symlink leads, so a confined session learns neither
which paths exist beyond its roots nor what its links point at. `agent_spawn` judges
`writable_roots` and `readable_roots` the same way. A command under Landlock alone and under
macOS can still tell a missing path from a denied one, as above.

On Windows the shell cannot hold this boundary: neither token mechanism confines reads. The file
tools keep it, a command does not, the agent is told so in its environment block, and startup logs
one warning when `confine_reads` is on.

If `[shell].sandbox = false`, `shell_execute` is refused at `workspace` rather than run
unconfined. Nothing else would be holding the boundary, and half a boundary reported as a whole one
is worse than an error that says so. Use `unrestricted` for those turns.

### What it does not cover

Four limits, stated plainly because none of them is visible from the inside:

- **MCP servers are not sandboxed.** They run in their own process, which meka does not confine, so
  a tool from an MCP server can write anywhere the server can, and no boundary meka can express
  reaches it. A tool with no permission annotation falls back to `unrestricted`, and meka **refuses**
  it at `workspace` rather than dispatching it, because `Permission::allows` treats `workspace` and
  `unrestricted` as equal and would otherwise let it straight through. To use one from `workspace`,
  name it in `[mcp.servers.*].tool_permissions` at a level you are willing to grant, turn approvals
  on so the call is put to you, or switch to `unrestricted`.
- **meka's own stores are outside the boundary and always writable**: the store under
  `MEKA_DATA_DIR`, memories included, and skills under `MEKA_CONFIG_DIR`. They are governed by their own
  config keys, not by this one.
- **Reads are not confined unless `[permissions].confine_reads` is on.** The boundary alone is
  "this cannot change things outside the workspace", not "this cannot see them"; the switch adds
  the second half, as [the read boundary](#the-read-boundary) describes.
- **The in-process fence resolves paths, it does not pin them.** `file_write` and `file_edit`
  resolve every existing component of a target before judging it, so a symlink already planted on
  the path is caught. What is left open is the race: a directory checked and then swapped for a
  symlink before the write lands. Closing it means holding a directory descriptor through the write
  on every platform, which is a larger mechanism than this one. It needs a *concurrent* writer
  planting the link mid-call to matter, which is consistent with the sandbox being defense against
  an agent damaging your data by accident rather than an adversarial containment boundary.

### Per-platform enforcement

| Platform | Backend | Confines the shell's writes | Confines the shell's reads, with `confine_reads` on |
|----------|---------|-----------------------------|-----------------------------------------------------|
| Linux | Bubblewrap (preferred) | Yes: read-only root bind, plus a writable bind per root, with the Landlock ruleset inside where the kernel has one | Yes: an empty root with only the readable paths bound into it, read-only, the roots read-write; what is not bound is absent |
| Linux | Landlock (fallback) | Yes: one path-beneath rule per root, or one per sibling where the root contains meka's directories | Yes: read and execute rules on the readable paths instead of on `/`; names and sizes outside stay visible to `stat` |
| macOS | `sandbox-exec` | Yes: writable subpath per root | Yes: data reads and executable mappings per readable path instead of everywhere; existence and metadata stay global so paths resolve |
| Windows | `WRITE_RESTRICTED` token + per-root ACE | Yes: writes are permitted only where a workspace capability has an ACE | No: the file tools hold the boundary, a command reads anywhere, and startup warns once |

Under Bubblewrap, `/tmp`, `/run` and `/var/tmp` are masked with a tmpfs, so paths there are not
merely unwritable but invisible. A workspace root under `/tmp` is bound after the mask and stays
reachable.

meka's own directories are hidden too: the config directory and the data directory holding
`meka.db` and every account credential. Bubblewrap masks them
after every workspace bind and `sandbox-exec` denies them last, so a confined command cannot read
the credential store even from a workspace root at `$HOME` that contains it, and the in-process
`file_write` and `file_edit` refuse a target under them whatever roots the session holds. The
in-process read tools (`file_read`, `file_search`, `file_find`, `scratchpad_load_file`) refuse
them below `unrestricted` too, and a search from a root above them steps around them. The one exception is meka's own skills store, `skills/` under the config directory, which those readers reach at every level: a skill's bundled files are what its body tells the agent to read, and nothing secret lives there. The same
readers refuse a process's own `/proc` directory (`/proc/<pid>`, `/proc/self`,
`/proc/thread-self`), whose `environ`, `cmdline`, `fd` and `mem` carry what the process was given,
meka's own environment and an MCP server's credential included; the system-wide files beside them
(`/proc/meminfo`, `/proc/cpuinfo`, `/proc/mounts`) stay readable. Landlock
rules only add access, so meka hides them there by never covering them: reading files is granted
per sibling along the path to each, which keeps their names and sizes visible and their bytes
not, and a workspace root above them is split the same way, at the cost that a new file under
such a root can land only in one of its subdirectories. The Windows token cannot express the
denial at all: under it a command at `read` can still read the config directory and the store,
account credentials included, and a workspace root containing them can write them; see
[Shell](../tools/shell.md#macos-and-windows). Only `unrestricted` writes there on the other
backends.

Windows works differently enough to be worth stating. meka mints a deterministic capability SID per
workspace root, adds an inheritable write ACE for it on that root, and runs the shell under a
`WRITE_RESTRICTED` token carrying that capability. Three consequences:

- **It writes to your directory's ACL.** The grant is real, standing state, visible in `icacls` as
  an `S-1-4-…` entry. meka takes it back when the process exits, including on Ctrl+C, and logs how
  to remove it by hand if revocation fails. The next run re-adds it, which costs one pass over the
  tree.

  **A crash or a kill still strands it.** Nothing runs on those paths, so the ACE outlives them. It
  grants nothing to anyone but a meka run in that same directory, and is reused rather than
  duplicated next time, but if you want it gone:
  `icacls "<root>" /remove:g *<the S-1-4-… from icacls>`.

  The grant is tracked per process, not per session, so several sessions confining the same root
  share one ACE, and it is released when the process exits rather than when any one of them ends.
  Under `meka serve` that means the ACE stands for the lifetime of the server.
- **It needs you to own the root.** Ownership supplies `WRITE_DAC` implicitly, which is what lets
  meka grant without elevation. A network share or another user's folder cannot be a workspace root.
- **Writes are restricted; nothing else is.** A `WRITE_RESTRICTED` token intersects write accesses
  only. Anything carrying an explicit `Everyone: Write` ACE stays writable even outside the
  workspace, which has no Unix analog.

  This one is a deliberate trade, not an oversight. The restricting list has to include `Everyone`
  or PowerShell cannot start: the .NET runtime fails to initialize with `E_ACCESSDENIED` before it
  evaluates anything, so every shell command dies. Measured both ways on Windows 11: dropping
  `Everyone` closes the hole and takes the entire shell with it. Writes inside the workspace, to
  files new and pre-existing, and to meka's own output pipe all behave the same either way, so a
  filesystem-only test makes the change look free. Files carrying an explicit `Everyone: Write` ACE
  are rare and usually a misconfiguration in their own right; a `workspace` level that cannot run a
  command is not a usable level.

- **The confined child shares meka's console and integrity level.** `read` gets a private
  console and a Low-integrity token, so Windows' UI privilege isolation stands between it and meka.
  A `workspace` child gets neither: a restricted token cannot create a console, only inherit one,
  and the integrity label is deliberately left alone so ordinary tooling keeps working. The child
  can therefore write to the terminal outside meka's own rendering, and window messages between the
  two are not blocked. It is confined on the filesystem, which is what the level promises, and it is
  not isolated from the meka process itself.

- **A `workspace` command can read meka's process memory, and a `read` command cannot.** This is
  the one axis on which `workspace` is *weaker* than the level below it, so it is worth stating
  plainly. `WRITE_RESTRICTED` intersects the restricting SIDs for write access only, and the
  integrity label is left alone, so nothing stops a `workspace` child calling `OpenProcess` with
  `PROCESS_VM_READ` against meka and reading whatever the process is holding, **including your
  account credentials**. Measured on real hardware: a native probe run at `workspace` read a
  canary string straight out of meka's heap, while the identical probe at `read` failed at
  `OpenProcess` with `ERROR_ACCESS_DENIED`, because Low integrity refuses the handle. There is no
  clean fix inside the current design. Dropping the `workspace` child to Low integrity would
  confine it to the Low-integrity surface and take the workspace write grant with it, and a deny
  ACE on meka's own process would have to name a SID the child carries but meka does not, which
  the restricted token does not provide. Treat `workspace` on Windows as protecting your files
  from the agent, not as protecting meka's secrets from a command the agent runs.

- **PowerShell runs in ConstrainedLanguage mode.** The restricted token triggers it, and `read` and
  `unrestricted` are unaffected (both report `FullLanguage`). Scripts that construct .NET types or
  set properties on them will fail at `workspace` where they work at `unrestricted`. meka's own
  UTF-8 output preamble is skipped rather than run there, so non-ASCII output at `workspace` is
  decoded with the host's legacy code page and may be mangled.

The mechanism is a port of a community proof-of-concept rather than a vendor-supported sandboxing
API, unlike Landlock, Bubblewrap and Seatbelt. It is the tightest boundary Windows offers without
provisioning machine-level identities, which would need an Administrator setup step.

## Default permission

The default permission is **read**, with approvals **off**. The default *enabled* set is every
level, `none / read / workspace / unrestricted`.

Shift+Tab reaches `workspace` before `unrestricted`, so the confined level is the one you land on
first when you want the agent to change something.

You can change the start level with:

- CLI flag: `meka --permission workspace`
- Environment variable: `export MEKA_PERMISSION=workspace`
- Config file: `[permissions] default = "workspace"`; see [Config file](../configuration/config-file.md#permissions)

If `--permission` or `MEKA_PERMISSION` selects a level that isn't in `[permissions].enabled`, meka
logs a warning and starts at the configured default instead of refusing to launch. A session whose
recorded level has since left `[permissions].enabled` is treated the same way wherever it is
reopened (a resume, `meka serve` re-attaching it, ACP `session/load`): it starts at the configured
default, with one warning naming the session, rather than at authority the configuration withdrew.

A level meka does not have in `[permissions].enabled` or `default` is refused at parse, with the
line. An `enabled` list that names nothing is different: meka warns and falls back to **`read`
alone**, not to the default set. An empty list asks for nothing, and answering it with four levels
including `unrestricted` would resolve to more authority than you wrote.

## Upgrading from `write`

The `write` level was split in 0.42 and the name is retired. It resolves to nothing, and every
surface says which of the two replaced it:

- **`workspace`** for writes confined to the working directory. This is what most `write` users
  actually wanted.
- **`unrestricted`** for the old behavior exactly: no boundary, no sandbox on the shell.

`write` is refused rather than reassigned on purpose. The same words are also *requirements* in
`[tools.tool_permissions]`, `[mcp.servers.*].tool_permissions` and `[mcp].default_permission`, where
silently re-pointing the name at the narrower level would have admitted tools a rung earlier than
their author intended. A hard failure at every door is the safe direction.

Anything meka persisted for itself (a sub-agent's saved spec, a scheduled job's gate) needs the
one-shot migration script; those values were never typed by you and cannot be fixed by hand.

## Changing permissions at runtime

Press **Shift+Tab** to cycle through permission levels:

```text
none → read → workspace → unrestricted → none → ...
```

Disabled levels are skipped during cycling.

Or use the `/permission` slash command:

```text
/permission workspace
/permission unrestricted
```

`/permission <level>` against a disabled level prints an error naming the currently enabled set.

The prompt indicator updates immediately to reflect the new level. The agent learns the current level via a per-turn `[Permission context]` block prepended to your message (see *How permissions work* below).

## Approvals

Approvals is one switch beside the level. Off, a call that needs more than the session's level is
refused, and the agent is told which level it would need. On, the call is paused for your approval
instead:

```text
• shell_execute
  $ ls -la
  Allow                  y
  Allow for the session  a
  Deny                   n
  Deny for the session
  enter to select · esc to close
```

The call is drawn as the transcript draws it, its arguments under its row. The picked answer is
highlighted; Up and Down move the pick.

It is off by default. Turn it on for a session with `/approvals on` on the screen, `approvals` on
`POST /v1/sessions` or `PATCH /v1/sessions/{id}` over HTTP, or the `approvals` config option in an
ACP client; set `approvals = true` under `[permissions]` to start every new session with it on. Like
the level, it is recorded on the session and comes back on a resume.

**An approved call still runs at the session's level.** Approval turns a refusal into a question; it
does not widen reach. An approved `file_write` at `read` lands only under the workspace roots, and
an approved `shell_execute` at `read` runs in the read-only sandbox. To let an approved call reach
further, raise the level. At `unrestricted` nothing sits above the level, so nothing is ever asked.
A call the level refuses however you answer is refused without a prompt: a write outside the
workspace roots, or `shell_execute` below `unrestricted` when nothing can sandbox it.

`none` with approvals on is the most cautious shape: every tool call is put to you, and nothing runs
unattended. Sub-agents share their parent's switch, and their prompts are forwarded to the parent's
frontend.

On the screen the prompt is a panel in place of the composer: the call as its title, every
argument under it, and four answers. Move the cursor with Up and Down and press **Enter**, or press
the answer's key: **y** allows, **n** denies, **a** allows for the session, and **1** to **4** pick
by position. **Esc** denies. If denied, the agent receives an error and may try an alternative
approach.

**Allow for the session** approves this call and every later call to the same tool for the rest of
the session without asking; **Deny for the session** denies them the same way. Both are keyed on
the tool, not on the arguments: allowing `shell_execute` for the session approves every shell
command the agent runs afterwards, so use it for the tools you trust wholesale and keep answering
`y` for the rest. A new session starts with nothing remembered, and `/fork` moves you into a new
session, so the answers stay with the one you branched from. The same two answers are ACP's
**Always allow** / **Always deny** options and the HTTP API's `allow_always` / `deny_always`
outcomes. The decision is one row on the transcript, `• shell_execute allowed for the session`.

Where nobody can be asked at all (a `--oneshot` run, or `meka serve` answering a turn with no
stream to put the prompt on), a call that needs approval is denied and a warning names the tool
that was refused without asking: as a `!` row on the screen, on stderr in a one-shot run, in `notices` on the JSON
surfaces. Without it a run whose every gated call was refused reads as a model that chose not to
use its tools.

Ctrl+C at the prompt cancels the turn and withdraws the question; the panel closes and the status
row says so. Esc at the prompt denies the call and lets the turn go on.

This is useful when you want the agent to be able to try things but want to review each action that
goes beyond the level before it executes.

### What the prompt shows

**Every argument the tool was called with**, not just the one the call's row picks out. That
distinction matters: the row's argument is the *destination* for every write-shaped tool, so a
prompt built from it would ask you to authorize writing to a path without showing the content, or
editing a file without showing the edit.

```text
• file_write
  │ path: src/auth.rs
  │ content:
  │   pub fn verify(token: &str) -> bool {
  │       true
  └   }
  Allow                  y
```

A long value wraps rather than being cut, so the end of a shell pipeline cannot be hidden from the
line you are approving.

**Nothing is left out.** A value too long for the panel wraps onto as many rows as it takes, and
the panel scrolls: Up, Down, PgUp and PgDn move through it, and the footer says how many rows are
below. A shell pipeline puts its consequence last, so a prompt that cut the value anywhere would
hide the exact part you are being asked about; a call with two hundred arguments is two hundred
arguments to scroll through, and that much to read is itself the signal to deny.

When a call is too much to read, **denying costs nothing**: say no, inspect the file or the
session with `meka session export`, and let the agent retry.

This is deliberately more than a call's row shows: the row is a notification, the prompt is a
decision, and a collapsed row does not make your approval prompts show less.

One consequence worth knowing: if the model passes a secret as a tool argument, an approval prompt
puts it on screen. That is the correct trade at the moment you are authorizing the call, but it does
mean such a value lands in your scrollback.

## How permissions work

When the agent attempts to use a tool, meka checks whether the current permission level allows it:

- If allowed, the tool executes normally.
- If not, and approvals are on, you are prompted to approve or deny.
- If not, and approvals are off, meka returns an error message to the agent explaining which level is required and suggests asking you to raise it.

### Telling the agent the current level

meka lists **every registered tool** in the per-turn `<context>` block with its required permission level inline (nothing is filtered out), and the same block carries a compact `[Permission context]` section:

```text
<context>
[Permission context]
Level: read. Approvals: off.

[Execution context]
Image input: enabled.

[Environment context]
Working directory: /home/you/project

[Available tools]
- **file_read** (level `read`)
- **file_write** (level `workspace`)
...
</context>
```

The section carries only the two values that change; what each level allows and what the approvals switch does are stated once in the system prompt. That short section is almost the only permission-dependent content in the request; `[Environment context]` is the other, since it is empty at `none` and gains a writable-roots block at `workspace`. The system prompt and the tools-array schemas stay byte-identical across `/permission` toggles, so mid-session level changes don't invalidate the Claude prompt cache; the entire conversation stays warm.

The same reasoning is why the tool catalog itself lives here rather than in the system prompt. Prompt caching is prefix-based, and the system prompt heads that prefix, so anything cached there that later changes (an MCP server connecting late or hot-swapping its tools, a skill being installed) would re-cache the entire conversation behind it. The `<context>` block rides inside your own message instead, so changes are appended rather than rewritten.

Only what actually changed is re-sent. The first turn of a session carries the full catalog, skill list, and any MCP server instructions; a turn where nothing moved carries none of it, and a turn where something moved carries a short note naming just that change.

### MCP tool permissions

MCP tools are classified through a 5-step resolution chain: per-tool override → server-level override → the server's own `readOnlyHint` → `[mcp].default_permission` → a hardcoded `unrestricted` fallback. See the *Permission resolution* section of the [Config file](../configuration/config-file.md) docs for the full rules and how to override a misclassified tool.

### Built-in tool permissions

Any built-in tool's required permission can be overridden from `config.toml` without editing code; see [`[tools]`: built-in tool filters](../configuration/config-file.md#tools-built-in-tool-filters). The same section documents how to allow-list or block-list specific built-ins (e.g. disabling `web_fetch` in a locked-down environment).

### Sub-agent permissions

Sub-agents spawned via `agent_spawn` inherit the parent's permission level by default. At `unrestricted` the sub-agent can call `file_write`, `file_edit`, and unsandboxed `shell_execute`; at `read` it's confined to read-only tools. To run one delegated task with reduced privileges, pass the `permission` parameter (e.g. `agent_spawn({prompt: "...", permission: "read"})`): it is clamped to the parent's level as a ceiling, so a sub-agent can only ever be equal-or-more restricted, never escalated. To narrow *where* it may write rather than whether, pass [`writable_roots`](../tools/overview.md#agent_spawn). A sub-agent shares its parent's approvals switch, and its prompts reach the parent's frontend. Alternatively, cycle the parent into a lower level before issuing the spawning prompt to restrict every sub-agent it spawns.

## Examples

### Read (the default)

```text
meka ~/project [r] > read the contents of main.rs
```

The agent uses `file_read` and shows the contents. Shell commands also work at `read`, but run in a **read-only sandbox**; the filesystem is write-protected for the child process:

```text
meka ~/project [r] > list the files in this directory
meka ~/project [r] > show me the git log
```

Commands like `ls`, `cat`, `git log`, `df`, `ps`, and `uname` work normally. Commands that attempt to write to the filesystem (e.g. `touch`, `rm`, `mkdir`) fail with a permission error.

Two things the sandbox deliberately does **not** restrict, on every backend:

- **Reads**, unless [`[permissions].confine_reads`](#the-read-boundary) is on. A sandboxed command can otherwise read anything your user can, including `~/.ssh` and `~/.aws/credentials`, and on Windows meka's own store. `read` protects the machine from being *changed*, not from being *read*.
- **The network.** Outbound connections are left open, so a command at `read` can still send what it read. Provider API keys are scrubbed from the child's environment, but that is one vector, not a boundary.

On Windows, `workspace` extends that first point to meka's own process. Its `WRITE_RESTRICTED` token restricts writes only, and unlike `read` it deliberately leaves the integrity label at the parent's level, so a confined command can open meka with `PROCESS_VM_READ` and read its memory. Measured on Windows 11: `OpenProcess` succeeds and `ReadProcessMemory` returns data. This is the one respect in which `workspace` confines less than `read`, whose Low-integrity token Windows blocks from opening a medium-integrity process at all. It grants nothing that reading `meka.db` would not, which a command at either level can already do, but it is worth knowing if you were treating `workspace` as strictly wider than `read` in every direction. They are not ordered that way; see the ladder note above.

If no sandbox backend is usable, shell commands at `read` **fail** rather than running unconfined. On Linux that means Bubblewrap (preferred whenever `bwrap` is installed) or Landlock at ABI v9 or newer. Landlock below v9 cannot refuse a `connect()` to a Unix socket on disk, so a "read-only" command could still reach D-Bus and `systemd-run --user` and have them act on its behalf, and meka refuses it rather than promise a protection the kernel is not enforcing. Kernels below 7.1 therefore need `bwrap` installed for the shell at `read`; `meka` says so at startup.

If you ask the agent to modify a file:

```text
meka ~/project [r] > add a comment to the top of main.rs
```

The agent will explain that it cannot write files at `read` and suggest switching to `workspace`.

#### What `read` still writes

`read` means the agent cannot modify **your tree**. It can still write to stores meka owns, because otherwise an agent at read permission could never remember anything:

| Store | Location | Tools |
|-------|----------|-------|
| Memory | the `memories` table in `MEKA_DATA_DIR` | `memory_write`, `memory_delete` |
| Skills | `~/.config/meka/skills/` | `skill_write`, `skill_delete` (only with [`[skills] agent_managed`](../configuration/config-file.md#skills)) |
| Scratchpad, scheduled jobs, background tasks | the store | various |

Of those, only the skill tools reach the filesystem at all; memory is a table in the store. Note what that makes `skill_write` at `read`: a persistence primitive. A skill it writes is read back into every later session's prompt, so a prompt-injected instruction can outlive the turn that carried it. That is the reason `[skills] agent_managed` is off by default and the tool is never given to a sub-agent. That boundary is enforced in two places: a skill name must be one path component matching the Agent Skills spec's own rule (lowercase letters, digits and hyphens), so it cannot contain `..` or a path separator, and a symlink sitting at that name is refused rather than followed, so an existing link cannot redirect a write out of the skills directory. Memory names are governed by a different and wider rule (`[A-Za-z0-9_-]`), which is safe for a different reason: a memory name is a primary key in a table, never a path. `file_write`, `file_edit` and `scratchpad_save_file` are the only built-ins that touch your tree, and all three require `workspace` or above, and are fenced to the workspace roots at that level.

A root you asked for is not always a root you get. `--writable-root` drops a path three ways, each with a warning: one that is not a directory, one naming a system directory the sandbox masks (`/`, `/proc`, `/dev`, `/sys`, `/run`, `/tmp`, `/var/tmp`, `$XDG_RUNTIME_DIR`), and one that does not resolve at startup; the last is kept rather than refused, so a build directory becomes a root the moment it exists. See [CLI options](../configuration/cli-options.md#--writable-root-path).

#### MCP tools are the exception

Tools from MCP servers are not built-ins and are not covered by that boundary. They execute inside the server's own process, which meka does not sandbox, so what an MCP tool may do is bounded by the server, not by meka's permission level.

What decides whether such a tool is reachable at read is the permission meka resolves for it, and by default a server's own `readOnlyHint: true` annotation is enough to classify it as `read`. That hint is asserted by the server and not verified. A server that advertises it for a tool that in fact writes therefore gets to write your tree while meka sits at `read`.

For a server you have not audited, either pin its tools explicitly with [`tool_permissions`](../configuration/config-file.md#mcp) or set [`trust_read_only_hint = false`](../configuration/config-file.md#mcp) on it, which makes the hint advisory for display only and drops its tools to the strict `unrestricted` fallback, past `[mcp].default_permission`.

So the honest statement of `read`'s filesystem guarantee is: your tree is safe from meka's built-in tools, plus whichever MCP servers you have chosen to trust.

> **Note:** The read-only sandbox uses Bubblewrap or Landlock (ABI v9+, kernel 7.1+) on Linux, `sandbox-exec` on macOS, and a Low-integrity token on Windows. See [Shell](../tools/shell.md#read-only-sandbox) for what each backend covers. Where no backend is usable, shell commands are not available at `read` or `workspace`. You can disable sandboxed shell execution by setting `sandbox = false` under `[shell]` in the config file (see [Config file](../configuration/config-file.md)), which makes `shell_execute` require `unrestricted` instead.

### Workspace

```text
meka ~/project [w] > run cargo test and show me the output
```

The agent uses `shell_execute` to run the tests and shows the results.
