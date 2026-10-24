# Interactive mode

Start meka without `--oneshot` to open the interactive screen:

```bash
meka
```

The screen takes over the terminal, the way an editor does. A new session opens on meka's mark,
the `M` of its icon in the terminal's cyan, two columns in, beside three lines: the version, the
profile the session runs on with its model, and the directory it opened in. It stands while
nothing but a warning stands with it, and goes with the first prompt. The
conversation fills the screen from the top;
the composer sits at the bottom, a gray block behind a `>`; and a status row under the composer
opens with the permission level, a cell of the level's color and its name in the same color, then
names the session's working directory (shortened with `~`), the profile the session runs on, the
context gauge (`128.4k/1.0M 13%`: tokens in context, the model's window, the share used), and
`2 background tasks` and `1 sub-agent` while any run.
[`display.status_row`](../configuration/config-file.md#displaystatus_row) picks the
facts after the level and their order, and can add the profile's model. Type and press **Enter**. What you sent stays on the transcript behind the same `>` in the same gray
block, in white, and the answer streams in under it as it arrives, word by word, settling into
rendered markdown as each line completes.

While the agent works, a row of its own appears above the composer, under the blank line that
always keeps the conversation off it: a spinner in column 0, then what the agent is doing
(`Working`, `Thinking`, `Answering` or `Using a tool`), how many seconds the turn has run, and
`Esc to stop`. The spinner stands still as `•` while the turn waits on you, for an approval or a
question, and the row goes when the turn ends. A `!command` has the same row, as `Running command`.

Everything on the transcript shares one left edge. Column 0 holds a glyph that says what a row is:
`>` what you sent, `•` something the agent or meka did (a tool call, a thinking block, a notice, a
command's outcome), `!` a warning, `■` an error, and nothing for the answer itself. No glyph is
an emoji, so none is drawn as a picture. The text of
every row starts in column 2. Color is spent sparingly: an answer's emphasis is weight (bold is
bold, headings are bold), the terminal's cyan names things (inline code, a tool call, the
checklist), structure (bullets, rules, the quote bar) is gray, and red and yellow mean an error
and a warning, nothing else. Fenced code is
highlighted from the same palette: comments gray, keywords blue, strings green, numbers yellow,
names of functions and types cyan. A code line wider than the screen folds at the edge, and a tab
is drawn as four columns. Every color is the terminal's own, so the screen follows your
terminal's theme.

```text
> summarize the project

• Thinking · The user wants a summary of the project state. I should look at…

• shell_execute ls -la && cat notes.txt
  └ total 12
    drwxr-xr-x 2 k4yt3x k4yt3x 4096 Oct 11 09:55 .
    … +3 lines
    gamma
• file_read missing.txt
  └ failed: failed to resolve path 'missing.txt'

• Checklist
  └ › Review the conflict section
    □ Write the summary

  Project state
  Here is a quick overview of where things stand.
```

The transcript is drawn from what meka knows, not written once and scrolled away, so what is on
screen can be changed after the fact: a thinking block or a tool call is a block that expands and
collapses, the whole transcript re-wraps when the window is resized, older turns load as you scroll
up, and you can type while the agent works.

## While a turn runs

Enter does not wait for the turn to end. What it does is the session's **send class**. While a turn
runs the empty composer names it: `Steer the agent`, `Queue a followup` or `Interrupt the agent`.

| Class | What Enter does |
|-------|-----------------|
| `steer` (default) | Posts the message to the session's inbox; the agent reads it at its next round, between two tool calls or after its reply |
| `followup` | Holds the message and sends it as the next turn once this one ends |
| `interrupt` | Cuts the answer being streamed and hands the agent the message at once |

**Alt+S** cycles the class for this run;
[`display.send_class`](../configuration/config-file.md#displaysend_class) sets where it starts. The
prompt appears in the transcript with its class and reads `waiting` until the agent has read it.
A slash command that would have to wait on the turn is refused while one runs; see
[Slash commands](#slash-commands).

The steer and interrupt classes are the inbox the HTTP API exposes; see
[the inbox](./http-api.md#the-inbox) for what each one does inside the turn.

When the turn ends, a row closes it, in the same dim register as the other stage directions:

```
• Finished at 19:28 after 19m 19s
```

The first word says how the turn ended (`Finished`, `Interrupted`, `Failed`), and
[`display.turn_end_summary`](../configuration/config-file.md#displayturn_end_summary) picks the
facts after it: the time, the duration, the tokens used. A resumed session closes every recorded
turn the same way, so the scrollback says when each one ended.

## Expanding and collapsing

Reasoning shows as one dimmed `• Thinking · …` row: the first words of the block once it has
ended, its latest line while it still streams, or the token count when the provider reports only
that. A tool call is its name with the one argument that identifies it, `• file_read notes.txt`,
and under a `└` the opening of what it produced: the lines up to the first blank one, at most
three, each cut at the edge, and a count of the rest, so a listing shows its header and first row
and a file its first lines, while what a tool wrote for the model after its payload is counted
rather than shown. A
call that failed says `failed:` in red there. A shell command is highlighted as code on the row.
A call that changed a file, a memory or a skill (`file_write`, `file_edit`, `memory_write`,
`memory_edit`, `skill_write`, `skill_edit` and the two deletes) draws the change as a diff in place
of the words the tool wrote: the line before, the line taken out in red with its old number, the
line put in in green with its new number, the line after, and `…` between hunks; its row counts the
change, `+1 -1`, and the arguments the diff already shows are left out of its expanded block. A
memory's diff is over the text `meka memory export` writes, so a reworded description is one changed
line of frontmatter, and a delete is every line in red. Collapsed, a change gets twenty rows rather
than the three of plain output, so a small edit shows whole and a rewrite shows its first rows and a
count. Either expands to the whole thing: the full reasoning, or every argument of the call wrapped
in full and all of its output. The row drops its summary once the arguments are under it; they hang
on a `│` bar from the row down to the `└` of the result, a shell command first, behind `$`,
highlighted and wrapped whole. Nothing in an expanded block is cut.

- **Click** a block to toggle it. A block lights up under the mouse, which is the hint that it
  opens.
- **Ctrl+T** shows or hides every thinking block at once; **Ctrl+O** expands or collapses every
  tool call.
- **Alt+Up** and **Alt+Down** pick a block out from the keyboard, newest first; **Enter** or
  **Space** toggles the picked one, and **Esc** lets go of it.

Everything else on the transcript is what it was when it happened: the prompts, the checklist as
it stood (the item in progress marked `›`, the rest `□`, a deferred one `»` with its reason),
the notices, the stage directions, and `! interrupted` where a turn was stopped. A slash command's
answer is not on the transcript at all; see [Slash commands](#slash-commands).

## Images

An image goes with the next prompt. Paste or drag a path to an image file (`png`, `jpg`, `gif`,
`webp` or `bmp`) onto the composer and the file is attached, or press **Ctrl+V** to attach the
image on the clipboard. meka reads that from the operating system, since a terminal pastes text
alone, so it needs a display on the machine meka runs on and does not work over SSH; a path does.
Each attachment is a chip above the text, **Backspace** on an empty composer drops the last one,
and **Enter** sends them with the words, shown as `• image` rows on the transcript. The profile has
to accept images (`[profiles.<name>].vision`), the limits are the HTTP API's, and an image waits
for the turn to end rather than riding on a steer.

## Scrolling and the mouse

The wheel scrolls the transcript; **PgUp** and **PgDn** scroll it half a screen, **Ctrl+Home** jumps
to the top of what is loaded and **Ctrl+End** to the end. The view follows new output while it is
at the bottom. Scrolling up pauses that, and `↓ back to bottom` appears in the row above the
composer, on the working row while the agent works, until you return: click it, press **Esc**, or
send a prompt.

Once an answer has pushed your prompt off the top, the prompt's first line sticks there as a bar.
Click it and the prompt comes to the top of the view, with the prompt before it stuck in the bar's
place, so clicking on walks back through the conversation one prompt at a time.

The screen opens on the last page of the session's log and reads the page above it whenever you
scroll near the top, so a resumed session shows its last turns at once and the first scroll up
reaches the ones before, however far back the session goes. What is held is bounded: scrolling far
up lets the newest pages go, and scrolling back down reads them again; returning to the bottom
always shows the end of the log with this run's rows after it. Older history shows what the log
holds, with a row for each compaction, in the words it was announced with, and each rewind.

The screen captures the mouse so that the wheel, a click and a drag reach it. Dragging over the
transcript selects the text it covers, and releasing copies it: to the terminal's own clipboard
through the OSC 52 sequence, which works over SSH and inside tmux where they allow it, and to the
system clipboard where meka can reach one. The status row says how many rows went. The terminal's
own selection still works with its modifier held (**Shift**; **Option** in iTerm2, **Fn** in
Terminal.app). Set [`display.mouse = false`](../configuration/config-file.md#displaymouse) to
leave the mouse to the terminal entirely; the keyboard still scrolls and toggles.

Inside tmux, the wheel needs tmux's own mouse mode (`set -g mouse on`).

## Keys

| Key | Action |
|-----|--------|
| Enter | Send, or post to the inbox while a turn runs |
| Alt+Enter, Ctrl+J, Shift+Enter | Insert a newline; Shift+Enter where the terminal speaks the kitty keyboard protocol |
| Shift+Tab | Cycle the permission level, skipping any not in `[permissions].enabled` (by default none &rarr; read &rarr; workspace &rarr; unrestricted &rarr; none) |
| Alt+S | Cycle the send class |
| Tab | Fill in the picked completion and keep typing; open the list again after Esc |
| Up, Down | Move the pick while the completion list is open; otherwise recall what you sent before, at the first and last line of the composer |
| Ctrl+R | Search what you sent before; Ctrl+R again steps to the next match |
| Ctrl+V | Attach the image on the clipboard to the next prompt |
| Ctrl+T, Ctrl+O | Show or hide every thinking block; expand or collapse every tool call |
| Alt+Up, Alt+Down | Pick a block; Enter or Space toggles it, Esc lets go |
| PgUp, PgDn, Ctrl+Home, Ctrl+End | Scroll the transcript; older turns load as you reach the top |
| Esc | Close the completion list or a panel (`q` closes a panel too); stop the turn, then the background tasks, then leave (see [Interrupting the agent](#interrupting-the-agent)); when nothing runs, back to the bottom, let go of a picked block |
| Ctrl+C | The same ladder as Esc while the agent works; clear the composer when idle, or climb the ladder from an empty one |
| Ctrl+D | Leave, when the composer is empty |
| Ctrl+L | Redraw the screen |

The composer takes the usual editing keys: Ctrl+A and Ctrl+E for the ends of the line, Ctrl+K and
Ctrl+U to kill to either end, Ctrl+W to kill a word, Alt+F and Alt+B to move by word, and the arrow
keys. It grows as you type, up to a third of the screen, and scrolls inside after that.

### Input history

What you send is saved to the store, so Up, Down and Ctrl+R recall what you typed in **any
previous run**. A brand-new `meka`, a resumed `meka -c`, and the current session all share one
history. Multi-line prompts are kept whole, and only the most recent entries are kept (older ones
are pruned). This input history is separate from the conversation itself.

## Slash commands

Commands start with `/`. `/help` shows this table, with the `/mcp` subcommands beneath `/mcp` and
the shortcuts below it. A command is not echoed on the transcript. What a command has to *show*
(`/help`, `/status`, a listing) opens in a **panel** in place of the composer: Up, Down, PgUp,
PgDn and the wheel scroll it, **Esc** or **Enter** closes it, and nothing of it stays on the
transcript. What the screen holds the answer to works at any time, a turn running or not: the
listings and pickers (`/skill`, `/memory`, `/schedule`, `/task`, `/session`, `/help`, `/log`), the
settings (`/permission`, `/approvals`, `/cd`), `/task cancel` and `/schedule cancel`, and `/exit`.
A command that needs the agent's turn to be over (`/compact`, `/rewind`, `/fork`, `/export`,
`/title`, `/status`, `/usage`, every `/mcp` verb and prompt, `/skill <name>`,
`/profile <name>`, `/project open <name>` and `/project close`) is refused while a turn runs, in a panel naming it: stop the turn with Esc
first, or wait for it to end. What a command *does* (set a level, change directory,
cancel a job) is one dimmed `•` row on the transcript. What a command has to *say back* instead, a
refusal, a failure or an empty list, is a panel as well, closed with Esc, and leaves no row.
`/log` opens the last few hundred log lines at every level, newest last, whatever `-v` lets onto
the transcript: a warning is a row as it happens, and the debug lines around it are there to read
without restarting.

A bare `/permission`, `/approvals`, `/profile`, `/project` or `/resume` is a **picker**: the choices in a panel with the
cursor on the current one, Up and Down (or **j** and **k**) to move it, PgUp and PgDn to move it by
a page, Ctrl+Home and Ctrl+End to the ends, **Enter** to apply, **Esc** to leave it as it was. In `/resume`'s list, **/** starts typing a filter that keeps only the rows
matching the text; **Esc** stops typing and keeps the filter, and a second **Esc** closes the list. With an argument each sets the value directly. A bare `/skill`, `/memory`, `/task` or
`/schedule` is a picker too: the skills and memories by name and description, the tasks and jobs by
id with what each is doing (a job that cannot fire is marked `held`), and **Enter** runs the picked
skill or shows the picked memory, task or job in full. A bare `/mcp` lists the configured servers
with their state, and **Enter** offers reconnect, login and logout for the picked one.

| Command | Description |
|---------|-------------|
| `/help` (or `/?`) | Show this help message |
| `/exit` (or `/quit`) | Leave |
| `/session` | Show the current session id |
| `/title [text]` | Show this session's title, or set it |
| `/permission [level]` | Pick or set the permission level |
| `/approvals [on\|off]` | Pick or set whether calls above the level are submitted for approval |
| `/profile [name]` | Pick or set the profile this session runs on |
| `/project [open <name>\|close]` | Pick, open or close the [project](./projects.md) this session is on |
| `/resume [session]` | Pick a session to switch to, or switch to one by id or unique prefix |
| `/compact [instructions]` | Summarize and compact the session, optionally saying what to keep |
| `/rewind [N]` | Drop the last N turns from the conversation (default 1) |
| `/export` | Export the current session as Markdown |
| `/fork` | Fork this session and continue in the copy |
| `/cd [path]` | Change working directory (bare: back to where meka started) |
| `/skill [name] [extra]` | Pick a skill to run, or run one by name with extra context |
| `/memory [name]` | Pick a memory to show, or show one by name |
| `/schedule [show\|cancel <id>]` | Pick a scheduled job to show; show or cancel one by id |
| `/task [show\|cancel <id>]` | Pick a background task to show; show or cancel one by id, or all |
| `/mcp <subcommand>` | Manage MCP servers and prompts |
| `/mcp list` | Pick a server to reconnect, log in or log out |
| `/mcp reconnect <server>` | Reconnect smoke-test for one server |
| `/mcp login <server>` | Run the OAuth flow for a server |
| `/mcp logout <server>` | Clear stored credentials for a server |
| `/mcp <server>:<prompt> [args]` | Render an MCP prompt as the next turn |
| `/status` | Show the profile, model, context use and cumulative session stats |
| `/log` | Show the recent log lines, at every level |
| `/usage` | Show account rate-limit usage (subscription backends) |

Some of the grammar the table compresses: a bare `/mcp` is `/mcp list`, and the picker shows each
server's live state (`pending` / `connected` / `failed` / `disabled`); `/task cancel all` is
accepted for `--all`; `/schedule cancel`, `/task cancel` and the two `show`s take an id or any
unique prefix; `/cd ~` still goes home; `/export` writes `session-<id>.md` in the working directory
and prints where it landed; `/skill <name>` prepends anything typed after the name to the skill body;
`/memory` lists memories most important first; `/mcp login <server>` puts the authorization URL
as a question in the composer's place, with one field for the callback URL pasted from the
browser's address bar when the browser cannot reach meka, and Esc on it ends the login.

Typing `/` opens the list of commands above the composer, each with its description, and the list
narrows as you type. **Up** and **Down** move the pick, **Enter** runs the picked command, **Tab**
fills it in so you can go on to an argument, and **Esc** closes the list until you type again. An
argument's list opens the same way: permission levels for `/permission`, `on`/`off` for
`/approvals`, configured profile names for `/profile`, `open` and `close` then configured project
names for `/project`, installed skill names for `/skill`, the
subcommands and configured servers for `/mcp`, and directory paths for `/cd` (Tab after a completed
directory drills into its subdirectories).

### `/status`

Shows the session's resolved model parameters followed by its cumulative counters, in a panel:

```
/status
  Profile:         work
  Account:         claude (claude-subscription)
  Model:           claude-opus-5-5
  Context:         128.4k / 1.0M (13% used, 871.6k left)
  Effort:          xhigh
  Thinking:        adaptive
  Turns:           23
  Compactions:     2
  Input tokens:    234.5k  (cache hit: 92%)
  Output tokens:   12.1k
  Redactions:      2 (12 images, ~38.0 MiB freed)
  Messages:        47
```

The top block reports what the session actually resolved to, in the order
[`[profiles.<name>]`](../configuration/config-file.md#accounts-and-profiles) declares the same
fields, so the two can be read side by side: the `Profile` with its account and backend, the `Model`, the
`Context` window, the reasoning `Effort` sent on the wire (omitted when nothing is sent, so the
provider applies its own default; `claude-subscription` sends `high` when the profile sets none),
and the `Thinking` mode. The rest are cumulative counters for the session.

`Context` is the live context-window occupancy: the total tokens of the most recent exchange (all input tiers plus output, i.e. what the next request re-sends minus your new prompt), against the active model's context window, with the percent used and tokens remaining. Use it to decide whether to `/compact` before continuing; after `/compact` it drops to the compacted size immediately. It reflects this session only; sub-agents spawned via `agent_spawn` have their own context and are not counted (a sub-agent's returned result is counted only once it lands in this session as a tool result). It is shown from the start, at `0 / <window>` before the first turn, since the window is your `context_window` setting (or the documented default) and this is where you confirm it took effect; it is omitted only when the window is unknown. The status row shows the same gauge after the profile; [`display.status_row`](../configuration/config-file.md#displaystatus_row) drops or moves it.

`Input tokens` (and the other cumulative counters) is the total billed across every turn of the whole session. These totals are persisted, so resuming a session with `meka -c` continues them rather than restarting at zero.

`Compactions` is how many times the conversation has been summarized, whether automatically, by `/compact` or at the agent's request. It is counted from the session's history, so it survives a resume, and each one puts the earliest detail one more summary away from the original.

`cache hit` is the share of input tokens served from the prompt cache rather than re-sent at full price, on every backend that reports one (Anthropic's cache tiers, OpenAI's `cached_tokens`). It should climb and stay high: meka keeps everything that changes mid-session out of the cached prefix, so a steady session re-reads the cache instead of rewriting it. The figure is arithmetic as much as it is health: every new token (a tool result, the model's own reply) is written to the cache once and read on each later request, so the session-wide share starts at zero, passes 90% around the fifteenth request (each tool call is a request) and keeps climbing from there; a low figure on a short session is not a miss. Expect it to drop once after a `/compact` (which rewrites the head of the conversation) and to recover on the following turns. A profile with [`cache_control = false`](../configuration/config-file.md#cache_control) on an endpoint that caches only what is marked reads zero here by design.

`Redactions` reports any times an Anthropic backend had to drop the oldest tool-result image blocks because the request body would have exceeded the profile's `max_request_bytes` (30 MiB by default, held under Anthropic's own 32 MiB limit). A non-zero count indicates the cache prefix was invalidated for the redacted messages.

### `/usage`

Fetches the account's current rate-limit usage from the account the session's profile bills and shows each rolling window with its percentage used and reset time, in a panel:

```
/usage
  5-hour (session)   [#---------]   8% used  (resets in 4h 12m, 2026-07-02 02:10 +02:00)
  Weekly             [----------]   2% used  (resets in 22h 50m, 2026-07-02 13:00 +02:00)
```

This is distinct from `/status`, which reports this session's own token counters. `/usage` queries the upstream service for your whole-account subscription limits. It works only for the two subscription backends, which expose a usage endpoint (`claude-subscription`'s 5-hour and weekly windows; `chatgpt-subscription`'s primary/secondary windows plus plan and credit balance). For an API-key backend it prints a short note that usage is not available there instead. The same command is available under ACP.

### `/compact`

The `/compact` command asks the LLM to summarize the entire conversation, then replaces the messages the model sees with a single summary message followed by the recent tail. This is useful for long sessions that are approaching the context window limit or becoming expensive.

After compacting, the session continues with the summary as context. The pre-compaction messages are never deleted: they stay in the underlying event log on disk (the model just no longer sees them). `meka session export` walks that full log, so an export always contains the entire conversation including the compacted-away turns, with a marker at each compaction point.

Every compaction, this command's and the automatic ones, leaves a dim row saying what set it going (`Compacted the conversation on user request.`, `Autocompacted the conversation after context usage crossed the ceiling.`, `The agent compacted the conversation.`), followed by what departed from the ordinary case: that no recent turns were kept, that the standalone summarizer wrote the summary rather than the agent, and the memories the checkpoint turn wrote. While one runs, the working row reads `Compacting the conversation`. A resumed session shows the same rows, read back from the log.

### `/rewind`

`/rewind` drops the most recent turn from the conversation, so the model no longer sees it or your prompt that started it. `/rewind N` drops the last `N`. The cut always lands on a turn boundary, so a tool call is never separated from its result. A compaction summary opens a turn of its own, so rewinding past everything after a compaction takes the summary with it and leaves the conversation empty; the turns it replaced stay behind the boundary on disk.

Like `/compact`, nothing is deleted: the dropped turns stay in the event log on disk, and `meka session export` still shows them with a marker where the rewind happened.

Use it to take back a prompt that sent the agent down the wrong path without paying for a summary, or to recover a session the provider has started rejecting. meka repairs a rejection it causes itself (see below), but content that entered the conversation earlier is out of its reach; rewinding past it is the way back. `meka session rewind <id>` does the same to a session you are not currently in.

### Recovering from a rejected message

Providers validate the whole conversation on every request, so one piece of content they reject would otherwise fail every later turn as well, permanently. When that happens, meka strips the offending content from what it added this turn, retries once, and hands the model the provider's own complaint as a failed tool result so it can adapt rather than silently losing the data. If the retry is rejected too, the original content goes back untouched and the turn reports the provider's error.

A mislabeled image already committed to the session is repaired when you resume it, without a provider round trip. For anything further back, use `/rewind`.

### Recovering from a call that got no answer

A refusal is one thing the provider says; a request that never got a usable reply at all is another. A connection that fails or is reset while the request is going out is retried with backoff (up to twice, waiting 1s then 2s), and so is a response body that could not be read back. The turn continues as if the failed attempt had not happened, and nothing about it enters the conversation. Only when the retries run out does the turn report the error.

Worth knowing what a retry can cost. When the failure was a body that could not be read, the provider had already generated the response and billed you for it, so the retry pays a second time. meka does it anyway, because the alternative is losing the turn for content you have already been charged for once, but it is not free.

One failure is not retried, because the next attempt is known not to be worth making: a request meka could not build, which points at whatever went into it, most often a `base_url` that is not a URL or a stored credential carrying a character that cannot go in a header. A redirect is never followed at all: the endpoint's answer is reported with where it pointed, and the fix is to set `base_url` to that endpoint. A reply, or one streamed event, larger than 64 MiB is refused for the same reason: an endpoint answering with something that size is not answering the request.

Retrying is bounded by time as well as by count, and the time bound is the one that usually decides. A failure that takes the full idle timeout to arrive costs five minutes, which spends the whole budget, so a stream that hung is reported rather than tried again: retrying is for a failure that was cheap, and a provider that went silent for five minutes has already taken more of your turn than a second silence is worth. Without the bound at all, three slow failures would be fifteen minutes of waiting on a turn that fails anyway.

The bound stops a *new* attempt starting rather than capping the total, so the worst case is a failure arriving just under the five minutes and permitting one more full-length attempt after it, for about ten in total.

### `/fork`

`/fork` copies the current session and switches you into the copy, printing its id. Your conversation carries over untouched, so the branch happens exactly where you are; the original stops there and keeps everything up to that point.

Use it before trying a direction you might want to back out of, or before `/compact` if you'd rather keep the uncompacted conversation around. To go back, exit and resume the original with `meka -r <old-id>`.

The copy is a fully independent session with no link back to its source. (`/fork` only ever runs
against the session you are in, which is never a sub-agent, so the sub-agent case below cannot arise
here.) See [Forking a session](./sessions.md#forking-a-session) for exactly what it carries.

## Shell escape

Type `!` into the empty composer and it enters **shell mode**: the `>` becomes a red `!`, the
status row drops its facts for a red `Run command in shell`, and what you type runs as a shell
command on **Enter**, bypassing the model entirely. **Backspace** on the empty composer leaves the mode, and a recalled
`!cmd` comes back in it. On the transcript the command keeps its `!`:

```text
> !ls -la
total 32
drwxr-xr-x  5 user user 4096 Mar  4 10:00 .
...
```

The command runs in the session's working directory, so it follows `/cd`. Its output is captured
and shown in the transcript, colors included, with a line naming the exit status when it failed.
It runs beside the screen, which keeps drawing and counts the seconds in the status row; **Esc** or
**Ctrl+C** kills it, with everything it started, and marks it `! interrupted`; so does leaving
meka, a hangup or SIGTERM included. (On Windows the shell alone is killed.) Its output keeps its
last 1 MiB and 10,000 lines, and a row says when more was let go. It reads nothing from the keyboard, and a command that needs the terminal itself (an editor, a pager) belongs in a
shell beside meka.

## Exiting

You can leave meka in any of these ways:

- Type `/exit` or `/quit`
- Type `exit` or `quit`
- Press **Ctrl+D** on an empty composer

Leaving while the agent is working stops the turn first, so nothing keeps running once the screen
is gone; a `!command` still running is killed, and background tasks are stopped on the way out,
with how many said.

The screen hands the terminal back as it found it. The session's id is printed on the way out
when [`display.show_session_id_on_exit`](../configuration/config-file.md#displayshow_session_id_on_exit)
says so; the conversation itself stays in the store, and `meka -c` reopens it.

## Interrupting the agent

Press **Esc** or **Ctrl+C** while the agent is running to interrupt it. The two keys count on one
ladder, and presses escalate:

1. The first press cancels the current turn: the request in flight is dropped and any shell command the turn spawned is killed. Background tasks keep running, because a keystroke aimed at the answer on screen should not lose a twenty-minute build.
2. A second press during the same turn stops every running background task, records each as canceled, and says how many it stopped.
3. A third press leaves, the way `/exit` does: what was canceled is given its moment to unwind, the background tasks are stopped, and the exit status is 130.

The count starts over with each new turn, so the first press of the next turn cancels that turn whatever happened during the last one, and it ends with the turn: a press during a wait that is not a turn (`/usage`, a stuck `/mcp reconnect`) starts a count of its own from the first step. When nothing is running, Esc goes back to the bottom of the transcript, and Ctrl+C clears a composer with text in it; on an empty one it counts on the ladder, so a second press stops the background tasks and a third leaves.

A question the agent asks replaces the composer until it is answered. An approval prompt is a
panel titled with the call, its arguments under it and four answers to pick from: **Allow** (`y`),
**Allow for the session** (`a`), **Deny** (`n`), **Deny for the session**, each also reachable by
its number; Esc denies. A server that asks for a URL to be opened asks the same way, with
**Open in the browser** (`y`), **Skip** (`s`) and **Decline** (`n`) to pick from. An MCP form
takes typed answers, one field at a time; Esc declines. While any of these is open, Esc answers
it rather than stopping the turn; Ctrl+C stops the turn, which withdraws the question. Questions
are put one at a time: when two tool calls dispatched together both ask, the second waits until
the first is answered.
