# Scheduling

Scheduling lets the agent arrange its own future work. Without it, Meka only ever acts when
something outside it asks: a human typing, an editor sending a prompt, a client calling the HTTP
API. A schedule is the one trigger nothing else supplies, which is what makes Meka usable as a
daemon or a standing assistant rather than a tool you drive.

The agent creates schedules itself through the `schedule_create` tool, so scheduling is usually a
conversation:

> **You:** remind me in 20 minutes to check the deploy
>
> **Meka:** Created schedule `7f3a1b2c` (once at 2026-08-11 15:22 +02:00). I'll remind you then.

## What a schedule is

A schedule pairs a **timing** with a **prompt**. When it fires, the prompt reaches its session as a
message: a turn already running reads it at its next pause, and an idle session gets a turn for it
(see [What a fire is](#what-a-fire-is)).

| Timing | Meaning | Example |
|--------|---------|---------|
| `at` | Fire once, then delete itself | `20m`, `2h`, `2026-08-12T09:00:00Z` |
| `every` | Fire on a fixed interval | `30m`, `1h`, `1d` |
| `cron` | Fire on a 5-field cron expression, in the schedule's `timezone` | `0 9 * * 1-5` |

Durations use the same syntax as `config.toml`, so `every = "30m"` means what
`[serve] idle_timeout = "30m"` means. Two things to know about it: **`m` is minutes and `M` is
months**, and decimals work (`1.5h` and `1h 30m` are the same duration).

Cron expressions have **no seconds field**, and follow standard Vixie semantics: when both
day-of-month and day-of-week are set, the schedule fires when **either** matches, not both. A six-field
expression is refused rather than read as Quartz, where `*/10 * * * * *` would mean every ten
seconds instead of the every-ten-minutes it looks like.

A cron pattern is read in the zone its schedule names in `timezone`, an IANA name such as
`Asia/Tokyo`, and in the machine's own zone when it names none, looked up at each fire, so a machine
that changes zone takes those schedules with it. The zone's own daylight-saving rules decide the instant:
`0 9 * * *` in `America/New_York` fires at 09:00 there on both sides of a change. `timezone` beside
`at` or `every` is refused, since neither reads one. Zone names come from the system's time-zone
database, kept current by the operating system; Windows, which has none, uses a copy built into
Meka.

A pattern that matches no calendar date (`0 0 30 2 *`) is refused when the schedule is created. One whose
next occurrence is far off is not: `0 0 29 2 *` waits up to four years for the next February 29th and
stays on the books until then.

## Gates: watching something without burning tokens

A plain recurring schedule spends a model request on every fire, whether or not anything happened.
Checking something every 15 minutes is roughly a hundred requests a day to say "nothing new"
ninety-odd times.

A **gate** is a cheap check run before the fire. Only if it says something happened does the
schedule fire. The interval then costs a tool call or a process spawn instead of a model call, which is what
makes a short cadence reasonable.

A gate has two halves. `check` is what to run, and `when` is what counts as "something happened":

```
schedule_create(
  every: "30s",
  gate: {
    check: { command: "gh pr checks 123 --json state -q '.[].state' | sort -u" },
    when: "changed"
  },
  prompt: "CI state for PR 123 changed. Investigate and report."
)
```

### What a gate can check

| `check` | Runs | Needs |
|---------|------|-------|
| `{ command: "..." }` | a shell command, unsandboxed | `unrestricted` |
| `{ tool: "name", arguments: {...} }` | a tool call, by the name the model uses | `read`, and the tool must resolve to `read` |

A tool gate is the one to reach for when a tool exists for the check. It is available at a far lower
permission, because a structured call to a server you configured is not a shell, and it returns
structured data that `when.at` can point into:

```
schedule_create(
  every: "1m",
  gate: {
    check: { tool: "mcp__mekabridge__unseen", arguments: {} },
    when: { at: "/chats", is: "not_empty" }
  },
  prompt: "There are unseen chats. Read them and reply if anything needs an answer."
)
```

**A gate may only call a tool that resolves to `read`.** A gate asks a question; a tool that can act
is not one. This is checked when the schedule is created *and* again every time it fires, so a tool that
resolves higher after an operator retunes it stops being a gate rather than
carrying on with authority nobody granted it.

### When a gate fires

| `when` | Fires when |
|--------|-----------|
| `"changed"` (default) | the whole result differs from the previous evaluation |
| `"succeeded"` | the command exits 0, or the tool call did not return an error |
| `{ matches: "<regex>" }` | the result matches the pattern |
| `{ at: "<json pointer>", is: "not_empty" \| "empty" \| "changed" }` | the pointed-at value satisfies the test |

One trap in the `"succeeded"` row: most MCP tools never set an error, so it is true on every
evaluation and the schedule fires every interval. It earns its place on a *shell* gate, where the exit
code is a real signal. For a tool, reach for `at` instead.

The gate's output goes to the model with the fire, so the model does not re-run the check the gate
just ran. It arrives under `[Gate output]` with terminal control characters removed and every
line quoted with `> `, so nothing the check read can pass for the fire's own header or a second
prompt. A pointer narrows what is *judged*, not what the model is told: the model still reads
the whole result, because the surrounding fields are usually what makes the fire worth reading. The
model reads at most 8 KiB of it, and `at` will read a document up to a megabyte; past that there is
nothing for a pointer to point into and the gate reports that the probe did not return JSON. A
command that prints past that megabyte is cut off there, stopped with everything it started if it
is still running, and does not count as `"succeeded"` whatever it exits with. A gate should be
reading a status, not a payload.

The two shapes that compare against the previous evaluation (`"changed"`, and `at` with
`is: "changed"`) always fire the first time: with nothing to compare against, "changed" is the
honest answer, and it means a typo surfaces immediately instead of lying quiet. The others judge the
result on its own, so a first evaluation is no different from any other: `"succeeded"` on a command
that exits non-zero does not fire, and neither does a `matches` whose pattern is absent.

**`changed` is only as good as the stability of the result.** The check should produce something
that changes when, and only when, the watched thing does, which is a stronger requirement than
"read-only" and is where most gates go wrong. It fails in both directions. A result carrying
something that moves on its own (a timestamp, an elapsed time, a request id, an unsorted list whose
order varies) differs on every evaluation, so the gate fires every tick and costs more than the
ungated schedule it replaced. A result that can return to an earlier value between polls (a bare count,
where two events arrive and one is consumed) reads as unchanged, and the gate silently misses what
happened in between.

This is the reason `at` exists. Almost any JSON result carries a field that moves on its own, so
`"changed"` over the whole of it is usually wrong; `{ at: "/chats", is: "changed" }` watches the one
field you mean and ignores the `checked_at` beside it. For a shell gate, pairing a count with a
monotonic marker (`git rev-list --count HEAD` alongside the commit sha) serves the same purpose.

> **A shell gate needs `unrestricted` permission.** It runs unattended, on a timer, until someone
> cancels it: a longer-lived grant than `shell_execute`, which at least ends with the turn that
> called it. It also runs with **no sandbox**, so `workspace` cannot authorize one: a level whose
> whole meaning is a write boundary must not hand out a command that has none. A *tool* gate is not
> held to this: `read` carries it, provided the tool resolves to `read` as well. Ungated reminders
> work at `read`.

> **`shell_execute` is a tool, and that is a door.** Where a sandbox backend is usable it resolves
> to `read`, so `check: { tool: "shell_execute", arguments: { command: "..." } }` is a legitimate
> tool gate at `read`: an arbitrary command, on a timer, from a session that could not have
> authorized the shell form. What makes that acceptable is that the two are not the same thing: a
> gate dispatches at `read`, which is the level `Confinement::resolve` sandboxes, so the command
> runs read-only-confined rather than as the bare `sh -c` a `command` gate would be. Where no
> sandbox is available the same tool resolves above `read` and the gate is refused instead, so
> "admitted" and "confined" cannot come apart. The confinement blocks writes, not the network: treat
> such a gate as something that can read this machine and talk to the internet, unattended, for as
> long as the schedule exists.

A gate that cannot run at all (it times out, the shell fails to start it, or its MCP server is not
connected) is **not** treated as "nothing happened". It is logged as a warning and the occurrence
is declined, because a watcher whose check broke otherwise looks exactly like a healthy watcher with
nothing to report. The marker that tells the *agent* about it needs two consecutive failures, and
those are now an occurrence or a lease apart rather than a poll interval, so a standing breakage
takes two periods to be reported rather than twenty seconds. That is the price of not re-running a
broken check at tick cadence.

Declined means *spent*, exactly as it does for a gate that ran and said no. A recurring schedule
moves to its next occurrence, so a six-hour schedule whose server is down is probed once every six
hours rather than on every poll tick. A one-shot has no next occurrence to move to, so it keeps its
claim instead and the retry waits out `[schedule] claim_lease` (an hour by default), long enough
that a server restarting near the schedule's due time does not cost the reminder, and bounded,
because each of those retries counts against the ceiling below. Either way the gate's stored
baseline is left alone, so when the check starts working again it compares against the last value
actually observed and reports the change that happened while it was broken.

A non-zero **exit code** is different, and only `succeeded` reads it as failure: for a large class
of good gates it is the signal. `diff -q a b` and `git diff --exit-code` exit 1 exactly when there
*is* a difference; `grep ERROR log` exits 1 through the entire quiet period it is watching; `curl -f`
exits non-zero until the endpoint returns. Every other predicate judges the output and logs the exit
code at debug level, so `-vv` will show you a command that is failing when you suspect one, without
a warning on every tick of the many gates for which a non-zero exit is the normal state.

## Where schedules run

Schedules belong to the session that created them, and are only checked while that session is live
in some Meka process. That makes the hosts behave differently, and the difference is worth knowing
before you rely on one:

| Host | Checks | Notes |
|------|--------|-------|
| `meka serve` | Every schedule, except on a session another process holds | Revives evicted sessions on demand. **The durable path.** |
| Screen | Only the schedules of the session it has open | Best-effort; a schedule goes dormant if you next start a different session |
| ACP | Only the schedules of sessions the editor has open | A fire appears in the transcript as a user message, where a turn read it |
| `--oneshot` | Never | The process exits; schedules stay on disk for a later run |

If you want a schedule to fire reliably whether or not you are sitting at a terminal, run
`meka serve`. There every fire is announced on the session's
[event feed](./http-api.md#the-session-feed) and to [webhooks](./http-api.md#webhooks) as
`schedule.fired`, naming the inbox item it is, so a client watching the session sees the agent's own
schedule at work as it happens. A turn opened on fires alone opens with a `turn.started` whose
`source` is `schedule`, with `item_ids` and `schedule_ids`; one opened on fires and a posted message
together says `inbox`; and a fire a running turn reads opens no turn at all, the feed's
`inbox.delivered` saying when the model read it. A message posted to the session's
[inbox](./http-api.md#the-inbox) while a fire's turn runs is read at that turn's next pause, and a
`followup` waits for it to end. On the screen, a schedule created in one session is checked only
while that session is the one on screen; `meka --continue` picks up where you left off.

Schedules all live in the same store, so a host that does not check a schedule has not lost it. A
schedule whose session nobody has open simply waits, and fires as soon as something that can run it
picks it up: another host, or a `meka serve` daemon pointed at the same data directory.

A schedule's fire always goes to the session that owns it, on every host. If what you want instead
is a recurring turn that carries no conversation at all, that is work for an external timer rather
than a schedule: systemd, cron or Task Scheduler invoking `meka --oneshot`, with the level and the
profile stated outright rather than inherited from a session.

```bash
meka --oneshot --permission read --profile work -p "summarize today's alerts"
```

### What a fire is

A schedule's check runs when it is due, whatever its session is doing, and its fire goes into the
session's [inbox](./http-api.md#the-inbox) as a `steer`, the class of a message that reaches a
running turn without cutting it short:

- **A running turn reads it at its next pause**: after a round of tool calls, before the next
  request, or after the reply that would have ended the turn, in which case the turn goes on to
  answer the fire, as it does for a steer or a finished background task. A client holding that turn
  gets both answers.
- **An idle session gets a turn for it at once**, from the host that holds the session: `meka
  serve`, `meka acp` for a session the editor has open, or the screen for the one on screen.

Whatever is waiting when a turn reads the inbox is read together, so several fires, or a fire and a
message, reach the model in one message. What the model reads of a fire is the schedule's prompt
under a header Meka writes:

```
[Schedule 6f9a0424 fired 2026-11-15 12:32 +00:00 while you were working]
[Late by 6s; this fire replaces 4 missed occurrence(s)]

Count the ticks.

[Gate output]
> 12:32:03
```

The header names the schedule by the first eight characters of its id and says when it fired, with
`while you were working` when a running turn read it. The second line is there only when the fire is
more than a minute late or stands for more than one occurrence, and `[Gate output]` only for a gated
schedule.

**A schedule has one waiting fire.** An occurrence that fires while the schedule's last fire is
still unread joins it rather than queueing another: the waiting fire then stands for one more
occurrence, carries the gate's latest output, and keeps its place in the inbox and any retry it is
waiting on. That is what the second line of the header counts. Under `meka serve` the joined fire
takes a new item id, and `schedule.fired`'s `replaces` names the one it had, which then answers
`404`.

**A fire is retried as any inbox item is.** When the turn opened on it fails before the model read
it, in a way waiting can change (the provider is unreachable, a quota ran out, a required MCP server
is not ready), its opening is taken back out of the context and the fire waits to be tried again:
after 10 seconds, then twice that per attempt up to five minutes between attempts, and it is given up
on an hour after it was queued (`[serve] inbox_retry_ceiling` under `meka serve`). Through a long
outage the waiting fire collects the schedule's later occurrences rather than piling up one
unanswered prompt per occurrence. A failure no retry would change, such as a rejected credential,
gives the fire up at once. The screen says `Trying schedule 6f9a0424's fire again at <time>` and
`Gave up on schedule 6f9a0424's fire: <reason>`; an editor is told the same words, after a warning
naming the schedule that failed; `meka serve` posts `inbox.deferred` and `inbox.failed`. A turn
that got as far as running a tool keeps everything, the fire included, since there is real work
behind it.

**A stopped turn withdraws what it opened on.** Stopping a turn opened on a fire (Esc or `/exit` on
the screen, `POST /v1/sessions/{id}/cancel`, `session/cancel`) withdraws the fire, as it withdraws any inbox
item the turn opened on and the model had not yet read. A turn stopped because its host is leaving
(`meka serve` or `meka acp` shutting down, a screen sent SIGTERM or SIGHUP or whose terminal goes
away) before the model answered takes its opening back and leaves the fire waiting in the inbox,
and the next host to hold the session opens a turn for it.

### Claiming an occurrence

A due schedule is *leased* before its check runs: the host records itself and an expiry on the row,
runs the gate, and then, in one write, spends the occurrence (advances the schedule, or retires a
one-shot) and puts the fire in the session's inbox. The claim covers the check alone; the turn that
answers the fire is the inbox's. Three consequences worth knowing:

- **A crash costs a retry, not the occurrence.** The row is untouched until the fire is queued, so
  a host that dies between claiming and queueing leaves a lease that expires, and the next host takes
  the occurrence. Once the fire is queued the claim is over, and what a crash leaves is the
  inbox's: a fire still waiting is read by the next host to hold the session, and one whose turn
  had written its opening when the host was killed stays in the conversation, where the model
  reads it at the session's next turn, whatever opens that; no turn is opened for it again.
- **A cancellation always wins.** Canceling deletes the row unconditionally, and a schedule canceled
  while its gate runs queues nothing. Canceling also withdraws every fire of the schedule the
  model has not read, so none is offered again and a turn that then fails does not try it again;
  one the model has read stays in the conversation. A fire whose text is already in the
  conversation cannot be unsaid, though: the next request a provider accepts carries it, be it
  the one already on its way or a later turn's, and the model reads the fire then; the item is
  `delivered`. A one-shot is retired as its fire is queued, so it leaves the list at once, and
  its fire is still delivered.
- **A schedule whose claims keep being abandoned is parked.** A claim that ends with nothing
  written, the occurrence neither spent nor its fire queued, is counted, and after three the
  schedule stops being retried. Two things abandon one: a host that stops, or a store that refuses
  the write, between the claim and the fire's write; and a one-shot whose gate probe cannot be
  evaluated, which keeps its claim so the retry waits out the lease. Both leave their claim to
  expire, so those three attempts are a lease apart rather than a tick apart: three crashes in half
  a minute would otherwise park a schedule whose only problem was a blip, and nothing retires a
  parked recurring schedule afterwards. The schedule is not deleted: it stays listed, cancelable,
  and marked as held with a reason that opens `3 claims ended without queuing a fire`, because a
  schedule that keeps taking its host down is something to look at rather than something to throw
  away. Recreate it, or cancel it. A host that simply declines a schedule it cannot take (`meka
  serve` finding the session locked by a screen) does not count, since that says nothing about the
  schedule. Nor does leaving the screen while a gate runs: the screen stops the gate and gives
  back that gate's claim as it goes, so the schedule is due again at the next start.

`[schedule] claim_lease` (default `"1h"`) is how long a lease is good for, and therefore how long a
crashed host's occurrence waits before another host takes it, and how long a one-shot whose gate
cannot be evaluated waits between tries. It should exceed a gate probe: a lease that expires under a
host still running the probe lets a second host take the same occurrence and run the probe again,
and the first host's fire is not queued. A host refuses to start on a `claim_lease` at or under
`gate_timeout`; a probe stopped at its timeout takes up to two seconds or so more to end its
processes and hand over its output, so leave some headroom on top of that check.

### More than one host on the same store

Several Meka processes pointed at one data directory (two `meka serve` instances, or a daemon and
a terminal) all poll the same table, so the same occurrence appears in several due lists at once.
Each occurrence is nevertheless checked once and fired once. A host takes it by leasing it in a
single conditional write, recording itself and an expiry on a row that no other host currently
holds, and the hosts that lose that write stop before evaluating the gate: no duplicate probe and no
duplicate fire. Which host wins is a race between their tickers and is not something you can pin
down; that only one wins is.

Two hosts sharing a session do not fight over it either. A session is held by one process at a
time, and `meka serve` leaves that session's schedules, and everything waiting in its inbox, to
whoever holds it: the holder checks the schedules, and its turns read the fires. `meka serve` does
not run the gate of a session it cannot serve, nor put off or count an attempt on an item it
cannot deliver; it takes the session up once the holder lets go.

Under ACP the editor is a live client, which changes one thing: approvals genuinely round-trip, so
a turn answering a fire can prompt you in the editor rather than being denied for want of anybody to
ask. Stopping it works the same as stopping any other turn.

When a fire reaches an idle screen, the turn it opens runs exactly like one you typed: output
streams, Esc or Ctrl+C stops it, and what you had half-typed stays in the composer. A stage
direction above the reply names the schedule and the opening of its prompt, `Schedule 1a2b3c4d
fired: remind me to stretch`, and says how late the fire is when it makes up for missed
occurrences: `Schedule 1a2b3c4d fired 3s late, replacing 2 missed occurrences: remind me to
stretch`, said as it happens and left out on a resume. A fire a running turn reads is the same
direction where it lands, below the tool calls of the round it followed and above the reply that
answers it. The prompt the model is given, with its header and any gate output, is not echoed, the
way no tool's output is; the reply is what you read.

## Restarts and missed occurrences

Schedules live in the store, so restarting the process (or the `meka serve` systemd unit) does not
lose them. What happens to a schedule whose time passed while Meka was down depends on the kind:

- **Recurring schedules fire once and resume.** A 30-second schedule that was down for six hours has
  720 missed occurrences; it fires once, and the model is told how many occurrences the fire stands
  for. It is then rescheduled from now, so an outage never turns into a burst.
- **One-shot schedules fire if they are still relevant.** Past `[schedule] missed_grace` (24 hours
  by default) they are dropped instead. A reminder to join a standup, delivered five days late, is
  noise. One that does fire is told how late it is, so the agent can judge whether it still matters.

That collapsing is per schedule. A session with several schedules due at once gets a fire from
each, and a turn reads every fire that is waiting at each of its pauses, so they are answered in
one turn rather than in a turn each.

If you want a backlog not to land at all, cancel the schedules (`meka schedule list`, then
`meka schedule cancel <id>`) before starting a host that will check them, or leave
`[schedule] enabled = false` while you clear it.

## Unattended turns and permissions

Under `meka serve` a turn opened on a fire has no request holding it, so with approvals on, a call
that needs approval is refused without asking, with a `notice` saying so, unless someone is there to
answer: a reader attending the session's feed with `attend=true`, or a session that declared
`answers_prompts_on_return` (see [Approvals](./http-api.md#approvals)). The schedule then fails to do
whatever needed approval. The denial appears in the session transcript rather than anywhere louder,
so a schedule on such a session that seems to do nothing is worth checking there first. A fire read
inside a turn a client holds is part of that turn, and its calls get that turn's approvals.

The screen and ACP both have someone attached, so approvals reach them normally.

A fire is answered at whatever permission the session holds when a turn reads it, not at the level
the schedule was created with: the level lives on the session and the session is mutable, through
Shift+Tab in the screen or `PATCH /v1/sessions/{id}` under `serve`.

With one floor: a session at `none` fires nothing, gated or not. Nothing is executable there, so the
turn would read nothing, change nothing, and could not even reach `schedule_cancel` to stop itself
being woken again. The agent can *see* the schedule (a tool's registration does not depend on the
permission level, so `[Scheduled]` still lists it and `schedule_cancel` is still offered) but every
call is refused at dispatch, which leaves it able to describe the problem and unable to fix it. An
`every = "5s"` reminder on such a session was a turn's worth of tokens every five seconds with no
in-session way to stop it. Raise the session to restore the schedule; it is declined, not canceled, and a
one-shot that came due while the session was down there is kept rather than spent.

A schedule's **gate** is the exception, because it runs unattended. Its bar is re-checked from two
places every time the schedule comes due: the level recorded on the schedule when it was authored, and the level the
session holds *now*. What that bar is depends on what the gate runs (`unrestricted` for a shell
command, `read` for a tool call), and for a tool gate the tool's own resolved level is looked up
again too, so retuning a tool's level takes effect on the next fire rather than whenever the schedule
is next rewritten. That level comes from `[tools.tool_permissions]` for a built-in, and for an MCP tool from the
five-step chain in [Permission resolution](../configuration/config-file.md#permission-resolution): the
server's `tool_permissions`, its `permission`, the tool's `readOnlyHint`, `[mcp] default_permission`,
then `unrestricted`. Step four is worth knowing about here: one global line turns every unannotated
tool on every server into a `read` probe a gate may call. And `readOnlyHint` is asserted by the
server and not verified by Meka, which is a weaker footing under a gate than under a call in
conversation, because nobody reads the result of a gate.

That second level is the session's own, recorded on its row and kept current by whichever surface
owns it: Shift+Tab and `/permission` on the screen, `session/set_mode` under ACP, `PATCH
/v1/sessions/{id}` under `serve`. The process that has the session open reads the level the session
holds right now, so a change takes effect there even if writing it to the row failed. Every other
process that polls the schedule reads the row, so withdrawing the level works across processes: a
`meka serve` daemon sharing the data directory will refuse a gate you just dropped in a screen.
Nothing else is read: every surface records a level when it creates a session, and a row that
somehow carries none fires nothing at all, rather than falling back to whatever the polling process
was started with.

Drop the session below what the gate needs and the gate stops running, and with it the schedule,
because a gate is the condition on the schedule and an unevaluated condition has not been met. The
occurrence is declined, and a warning is logged naming the schedule. Raise the session back to restore it. Unlike a gate
that ran and said "nothing happened", a held gate was never evaluated at all, so a one-shot that came
due while it was held is kept rather than spent.

The agent is told too, not just the log. A schedule that cannot currently fire is marked in the
`[Scheduled]` block it sees every turn and in `schedule_list`, as `NOT FIRING: <reason>`, with the
same sentence the warning carries; the moment a gate is withdrawn or restored is announced as a
world change. This matters because the two states are otherwise identical from the agent's side: a
held schedule and a healthy watcher with nothing to report both simply never fire. It can act on the
difference, since `schedule_cancel` needs only `read`.

A gate whose *probe* keeps breaking is marked the same way, after two consecutive failures. A server
that changed its schema, a command that was uninstalled, a pointer into a result that stopped being
JSON: each errors on every evaluation, and each is a dead watcher that looks exactly like a quiet
one. The first failure is deliberately not reported, because one failure is as often a blip as a
break. This one is tracked in memory rather than on the row, so it is known to the process running
the schedule: a restart re-establishes it within two poll intervals, and `meka schedule list`, which is a
separate process, does not see it.

Four surfaces report it, and each says only what it can establish:

- **`[Scheduled]` and `schedule_list`** carry the full sentence, since the agent is the one that can
  recreate or cancel the schedule.
- **`/schedule`** marks a schedule `held` when it cannot fire and `held?` when this process cannot
  establish the answer; no mark means "it will fire", not "I did not check". It shows `held?` for a
  schedule whose session level it could not read, since that is unestablished rather than fine.
- **`meka schedule show`** spells the same verdict out on a `withheld:` line. It is a separate
  process from any host and so cannot resolve a tool gate, reporting that as unknown rather than as
  fine. `meka schedule list` does not carry it at all: a column that is blank on almost every row is
  a poor use of a table this wide.

  Both apply `[permissions].enabled` when reading a session's recorded level, so neither can report
  a schedule as able to fire that the host refuses.
- **`GET /v1/schedules`**, with or without `?session=`, carries a `withheld` field with the same
  sentence, absent when the schedule can fire.

Firing the reminder ungated instead would be the more forgiving-looking choice and the wrong one: it
turns a conditional schedule into an unconditional one, so an `every = "1m"` watcher that normally
speaks once a week would deliver a fire a minute for as long as the session stayed below that bar. An
*ungated* schedule is unaffected by a gate's authority, and keeps firing at any level above `none`.

At `none` nothing fires at all, gated or not. Every tool is refused at dispatch there, so the turn
would read nothing, change nothing, and be unable to reach `schedule_cancel` to stop itself being
woken again: tokens spent to produce an agent that can describe its predicament and do nothing
about it. `POST /v1/schedules` refuses to create a schedule on such a session for the
same reason; `schedule_create` needs `read` to dispatch at all, so the agent cannot reach it.

Both halves are load-bearing. The recorded level rarely refuses on its own, since a creation door
already demanded it; the live level is what makes a withdrawal real. The recorded level still matters
for a schedule created before it was stored, which reads as "no authority" and stays refused.

## Inspecting schedules

The agent sees a short index of the current session's schedules in its per-turn context, so it can
avoid scheduling a duplicate. For details it calls `schedule_list`.

From your side:

```bash
meka schedule list                  # every session's schedules
meka schedule list --session 0b5c   # one session, by id or unique prefix
meka schedule show 7f3a1b2c         # one schedule in full, by id or unique prefix
meka schedule cancel 7f3a1b2c       # by id, or any unique prefix
meka schedule list --format json    # {"schedules": [...]}, the shape GET /v1/schedules answers with
```

`list` is a table to scan: schedule id, the session it fires in, its timing, how long until it next
fires, whether it is gated (`shell`, `tool`, or `-`), and the beginning of its prompt. Every cell is
bounded so the table stays legible. Under `--format json`, `list` and `show` print each schedule
with the fields [`GET /v1/schedules`](http-api.md#schedules) uses (`id`, `session_id`, `timing`,
`prompt`, `gate`, `created_at`, `last_fired_at`, `next_fire_at`, `withheld`), ids in full and nothing
truncated; the gate's `check` is always given, since the operator at the terminal can read every
schedule in the store.

Both ids print as a UUID's first segment, and widen only if that would show two rows the same
string, so what you see is normally enough to retype into `show`, `cancel` or `--session`, which
take any unique prefix. `show` prints both in full.

Normally, because uniqueness is computed over the rows being printed while `show` and `cancel` scan
every schedule there is. `list --session <id>` narrows the table, so it can print a prefix that
another session's schedule makes ambiguous. It fails closed (the command refuses and names the ids
that collided), and an unfiltered `meka schedule list` always prints a prefix that resolves.

A schedule this build cannot read (a row a newer Meka wrote, or one edited by hand) is left out of
every listing, with a warning naming its id; `cancel` takes that id, or a unique prefix of it.

`show` is the one that answers what a schedule actually does: the whole prompt, the whole command or
tool a gate runs, the session's full id, when it last fired, and whether it is withheld. Nothing
there is truncated, which is why it is a separate command rather than a wider table.

A fire that is waiting is not a schedule's: it is an item in its session's inbox, listed by
[`GET /v1/sessions/{id}/inbox`](http-api.md#the-inbox) with its `schedule_id` and the number of
`occurrences` it stands for. Canceling the schedule, by any of these doors, withdraws it.

On the screen, `/schedule` is a picker of the current session's schedules, each with its timing and
prompt, and Enter (or `/schedule show <id>`) shows one in full; `/schedule cancel <id>` cancels one.
All three answer inside the conversation you are in: a schedule belonging to another session is not
found here, which is what makes the ids the picker shows the ids the other two take.

## Configuration

```toml
[schedule]
enabled = true          # default true; false hides the tools and stops the scheduler
poll_interval = "10s"   # how often due schedules are checked
missed_grace = "24h"    # how late a one-shot may be and still fire
gate_timeout = "30s"    # wall-clock budget for a gate probe
max_schedules = 50      # per-session ceiling, refused at schedule_create
claim_lease = "1h"      # how long a host's claim on an occurrence is good for
```

`poll_interval` is the real resolution floor: a schedule with a shorter interval fires once per
tick, not once per interval.

Setting `enabled = false` keeps the three `schedule_*` tool schemas out of every request and leaves
existing schedules on disk without firing. `POST /v1/schedules` refuses with a 404 rather than
accepting a schedule that could never run; `GET /v1/schedules` and
`DELETE /v1/schedules/{schedule_id}` keep working, so schedules left over from before the flag was
flipped can still be listed and cleared.

## Tips

- Write the prompt for a reader who has no context. The conversation that created the schedule may
  be long over, and after a compaction the turn that created it may not have survived. A fire can
  also arrive in the middle of other work, read between two tool calls of a turn about something
  else, so the prompt should say what to do without leaning on what the agent was doing when it was
  written.
- Reach for a gate whenever the answer is usually "nothing happened".
- Keep gate probes fast and read-only. They run on every occurrence, whatever the session is doing,
  and a gate that changes something is a side effect on a timer.
- Check what a gate's probe returns across two runs where nothing happened. Identical output is the
  whole mechanism, and anything varying inside it turns the gate into a timer.
- If a schedule matters, check what `schedule_create` reports back: it states the resolved next fire
  in absolute local time, which is how you catch a cron expression that parsed fine and means
  something other than you intended.
