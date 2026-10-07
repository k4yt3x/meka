# Checklist

The checklist is the list the agent keeps of what it has committed to do, and the rule that a turn
cannot end while an item on it is open. The agent adds items as it commits to them, checks them
off as it finishes them, and if it tries to stop with an item still open, meka sends it back to
the list instead of ending the turn.

It exists for one failure: an agent that plans five steps, does three, and reports back as if it
were done. The list is the agent's own words, so meka never has to judge the work; it only has to
hold the agent to what it said.

## How it works

The agent has three tools:

- `checklist_add` appends items. Each is a text, or an object `{"text": ..., "status": ...}` with
  `in_progress` for the one it is starting on. The call returns the ids assigned and the whole
  list.
- `checklist_edit` changes one item by `id`: its `status`, its `reason`, the background `task` a
  deferred item waits on, or its `text`. A call with anything wrong in it is refused whole, a
  task that has already reported included.
- `checklist_read` returns the list.

An item sits in one of three states: `pending`, `in_progress`, or `deferred`. Two more words take
it off the list: `completed`, and `canceled`, which needs a `reason`. Completed and canceled items
leave the list at once; the conversation keeps what was done and why.

**A turn cannot end while an item is pending or in progress.** When the agent replies without
calling a tool and such an item remains, meka appends a message listing the open items and the
ways out, and the agent answers again. It can keep working, complete the items, cancel them with
a reason, or defer them with a reason. The turn ends once every item is disposed of. Each nudge is
announced as a notice, so a turn that keeps working after its answer does not look like a hang:
on the REPL as a stage direction between the reply and the next round, `(checklist: 1 open
item, continuing, nudge 1 of 3)`, and on the HTTP feed as a `notice` with the same text.

**Deferral is the honest way to stop.** A deferred item does not hold the turn. It is for an item
that cannot proceed now because it waits on a person, an event, or an explicit "later", never
because it is hard or tedious; the tool descriptions and the nudge both say so, and the reason is
recorded. A deferred item stays deferred until the agent or you pick it up again: meka does not
guess when a person has answered. The one case it can see is a background task. An item deferred
with the `task` it waits on reopens once the task's outcome has been reported, however the task
ended; the report arrives as a turn of its own, and that turn cannot end without the agent
finishing, canceling or re-deferring the item. A task that has exited but not yet reported holds
nothing, so the turn that deferred on it ends normally even when the task ended under it, and
the report's turn is the one held.

A task that has already reported cannot be waited on: `checklist_edit` refuses a deferral that
names one, or that would keep one from an earlier deferral, because the item would be open the
moment it was deferred. A deferral states what the item waits on, so a re-deferral with a new
`reason` states it afresh, and a task not named again is dropped with the old reason; that is how
one call moves an item off a task that has reported. A re-deferral that gives neither keeps both.

A message you send mid-turn, a steer over the HTTP inbox or a parent's `agent_steer`, rides the
nudge the way it rides a tool round's results: the model reads it with the list rather than after
the next tool call.

**The cap.** An agent that answers the nudge three times in a row without calling a tool is not
going to on the fourth. After three such nudges the turn ends with the items where they are, and a
warning notice says how many were left open. A tool call between nudges starts the count over, but
not without limit: a turn gets nine nudges in all, so an agent that reads the list before every
reply and changes nothing is not sent back to it forever, and the notice then says `after 9 nudges
this turn`. The next turn's end asks again.

The nudge fires only on a normal end of the turn. A canceled turn, a failed one, a stop at the
output token limit and a refusal end on their own terms, with the list untouched.

## What you see

The REPL prints the list whenever it changes, one row per item: the one in progress yellow, the
rest grey, a deferred one marked `[>]` without its reason, which is a paragraph as often as not
and is in the record, the feed and `checklist_read` for whoever asks. A session record
over the HTTP API carries `checklist`, the open items, on a session the server holds, and the
session feed announces every change as `checklist.updated`. An ACP client shows the list in its
plan panel; a deferred item is pending there, with what it waits on in its text.

Nothing is injected into the agent's context between turns. It sees the list in every tool call's
result, in the nudge when it tries to stop, and through `checklist_read` whenever it wants to.
Compaction is the one exception: the open list is copied verbatim into the summary, because the
calls that built it are what the summary replaces.

## Where it lives

The list is recovered from the conversation itself. Every `checklist_add` and `checklist_edit`
call is a row of the session's history, and the list is what those calls add up to, so a resumed
session still owes its items, a rewind that drops a turn drops the items that turn added, and a
fork or an archive carries the list with the conversation. A compaction records the list on its
boundary, so a session that resumes from one starts from the list as it stood. Background tasks
are the session's own and do not travel: in a fork or an import, an item deferred on one stays
deferred until the agent or you pick it up, because no report of that task will ever reach it.

A sub-agent has a checklist of its own, held to the same rule on its own turns, and keeps it
across an `agent_followup`. Its list and its nudges are its own business: they reach its own feed
over the HTTP API and nothing of the parent's, so the parent's list on screen is always the
parent's. A sub-agent has no background tasks, so its deferrals carry a reason alone.

## Turning it off

Filtering the three tools out through the
[`[tools]` filters](../configuration/config-file.md#tools-built-in-tool-filters) disables the
feature: with no tools there is no list, and with no list there is no nudge. There is no other
switch.
