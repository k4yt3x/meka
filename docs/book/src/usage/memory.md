# Memory

Memory is the agent's own set of durable notes. It writes them itself, they survive compaction, and they outlive any single session.

Without it, an agent's only state is its context window. When a long session compacts, detail is summarized away; `conversation_search` can still search the message log, but only for something you remember to look for. Memory is the deliberate half: a fact the agent decided was worth keeping, in a store later sessions can search.

## How memory works

- Memories are rows in the `memories` table of the store (`~/.local/share/meka/meka.db`, or under `MEKA_DATA_DIR`), one row per memory.
- The store is scoped to the **meka instance**, not to a session or a directory. Everything sharing a `MEKA_DATA_DIR` shares one memory; pointing a deployment at its own data dir gives it its own. Within it, a memory is global or a [project](./projects.md)'s; see [Tiers](#tiers).
- The per-turn context includes a bounded memory index, updated when it changes. A memory's `load` decides how it gets there: an `inline` body is rendered whole, within a budget every inline body shares, an `index` memory is listed by its description, and a `search` memory is reached by `memory_search` alone. Other bodies are read on demand with `memory_read`.
- The index is re-stated in full at the start of a session, after every compaction, and whenever it scrolls out of the context window. This is what makes memory survive compaction.
- The agent is told to save durable preferences, constraints, decisions, and facts, to keep temporary task state in the scratchpad, and that current instructions and corrections take precedence over stored preferences.
- Memories are available at every permission level except **none**; all five memory tools ask only for **read**. Writing a memory therefore needs no write authority over your files, and `workspace`'s boundary does not apply to it: the store belongs to meka, not to your working tree.

> **Memories live in `MEKA_DATA_DIR`**, alongside sessions, rather than in the config directory. A backup of your config directory does not capture them; `meka memory export` is what does.

## Tiers

A memory is in one of two tiers: **global**, which every session sees, or a **project's**, which the sessions on that [project](./projects.md) see beside the global ones. A session on no project sees the global tier alone. Which tier a session is on is its project, set with `--project`, `/project`, or the session's `project` field over HTTP and ACP; a project is a row `meka project add` creates, and a name no project has is refused at every one of those doors.

- `memory_write` files a memory in the session's project unless `tier: "global"` says otherwise. In a session on no project every write is global, and `tier: "project"` is refused. Writing to a memory that exists keeps its tier unless `tier` names the other one, which moves it: that is how the agent re-files a note it saved in the wrong tier, and the result line says `global` when a project session's write lands in the global tier.
- A name is unique within a tier, and a global name is never also a project's, so a session never sees two memories of one name; two projects may hold the same name. A write that would make a global memory and a project memory share a name is refused with `choose another name`, at every door.
- The [inline budget](#the-inline-budget) is measured as a session's band sees it: a project write counts the global inline bodies and the project's, and a global write counts the global bodies and the fullest project's, since a global memory is in every project's band.
- The `[Memory]` section a session on a project reads is grouped, the global tier first and the project's after it, under one index budget. A move between projects restates the whole section under a line naming the project, and in a project session a change to a global memory is marked `(global)` in the per-turn delta, as a global hit is in `memory_search`'s results. `memory_search` takes `tier` to search one tier.
- The operator's doors see every tier: `meka memory list` shows a `Project` column, `--project <name>` and `?project=<name>` pick one tier (an empty value, the global tier), and a name that is in several projects is refused by `get`, `show`, `edit`, `remove` and the single-memory HTTP routes until one is named. `meka project rename` moves a project's memories with it; `meka project remove` refuses while any remain.

## Fields

| Field | Meaning |
|---|---|
| `name` | Unique identifier, `[A-Za-z0-9_-]`. Case-insensitive: `NOTE` and `note` are one memory. |
| `description` | One line used in the bounded index. Make it stand on its own. |
| `priority` | `0`–`9`, default `5`: how important the note is, not whether it loads. See [Priority](#priority). |
| `load` | How the memory reaches the context: `inline`, `index` (default) or `search`. See [Loading](#loading). |
| `tags` | Lowercase labels (`[a-z0-9-]`, at most 10) for grouping and filtering. |
| `body` | Detail, loaded on demand by `memory_read`, or in context every turn when `load` is `inline`. |
| `references` | What the memory depends on: memory names, paths, URLs, external ids. See [References and backlinks](#references-and-backlinks). |
| `source` | One line saying where the fact came from. See [Source and verification](#source-and-verification). |
| `created` | When the memory was made. Stamped once, at creation. |
| `updated` | When the row last changed. |
| `verified` | When the fact was last verified. Stamped at creation and moved only by `verified: true`. |
| `expires` | When the memory stops applying, if a date was set. See [Expiry](#expiry). |
| `read count` | How many times `memory_read` has opened it. Feeds search ranking. Only `memory_read` increments it: a search hit is weaker evidence, and reading through the CLI or the HTTP API is the operator rather than the agent. |

### `created` versus `updated`

These answer different questions, and conflating them was a bug. A `memory_write` that changes only a description or a priority moves `updated`, and reading that as the observation date made a years-old note render as "today", sort to the top of its priority band, and arrive through `memory_read` captioned "Saved today. This is what you recorded then".

`created` is stamped once, when the memory is created, and carried forward untouched by every later write. It is what the index renders as an age, what ties are broken by, and what freshness weighting reads. `updated` is reported by `meka memory get` and the HTTP API and takes no part in ordering or ranking.

The rule is enforced by the `INSERT ... ON CONFLICT DO UPDATE` statement itself, which never assigns `created_at` on the update path, rather than by each write door remembering to preserve it.

### Omitting a field keeps what is there

`memory_write`'s `body`, `tags`, `priority`, `load`, `references`, `source` and `expires` are all optional, and omitting any of them **keeps whatever the memory already had**. That makes a metadata-only update (a reworded description, say) a single call that cannot cost the note its contents, its labels, its rank, its dependencies or its deadline. To clear the body or the expiry, pass `""`; to clear a list, pass `[]`. A blank in any other field is the field omitted.

`PUT /v1/memory/{name}` and `meka memory add <name> --force` follow the same rule.

## Editing a memory

`memory_edit` changes part of a body the way `file_edit` changes part of a file: it finds `old_string` exactly once and replaces it with `new_string`, refuses a match that is ambiguous unless `replace_all` is true, and deletes the text when `new_string` is empty. It exists because `memory_write` replaces the whole body, and a long-lived note rewritten whole on every small change drifts: wording moves, lists reorder, and a line is lost with nothing said. An edit touches the text it was told to and nothing else, and the result shows the lines around the change. Pass `verified: true` on an edit that follows a check, and it verifies the memory as a write would.

A rewrite computed against a body that changed meanwhile is refused rather than landing over it, the same compare-and-swap `meka memory edit` uses when `$EDITOR` comes back.

## Priority

**Lower means more important** (the same direction as `nice`, the opposite of CSS `z-index`). Priority decides three things: where a memory sits in the index, which memories survive when the index hits its size budget, and how heavily search weighs it. It does not decide whether a memory's body is in context; that is [`load`](#loading), and a rule can be in front of the agent every turn without being its most important note.

| Range | Use for |
|---|---|
| 0–1 | Standing directives that always apply |
| 2–4 | Durable facts |
| 5 | Default |
| 6–9 | Situational or short-lived notes |

Within one priority band, the most recently *recorded* memory sorts first, so a fresh note never displaces a standing rule just for being new.

Because the agent picks a priority at write time and everything feels important then, priorities tend to drift downward over a long-lived instance. The Priority column of `meka memory list` shows that happening so you can rebalance. Search ranking compensates for the same drift from the other side: see [Search](#search).

## Loading

`load` says how a memory reaches the context, independently of how important it is:

| Value | What the per-turn context carries |
|---|---|
| `inline` | The whole body, every turn. For a standing directive the body is the directive, and leaving it behind a tool call means the agent has to look the rule up before it can follow it. Inline bodies share one budget; see [The inline budget](#the-inline-budget). |
| `index` | The description, as one line of the index. The default. |
| `search` | Nothing. The memory is reached by `memory_search` alone: a runbook, a reference note, anything worth keeping that is not worth a line every turn. |

The index ends with a count of the memories it lists nowhere, the search-only ones and the [expired](#expiry) ones, so the agent knows there is more than it can see. On upgrade from a release before `load`, every priority-0 memory becomes `inline` and everything else `index`, which is what the priority used to mean.

## The inline budget

Inline bodies share one budget, [`[memory] inline_max_bytes`](../configuration/config-file.md#memory), 32 KiB unless set. It is counted in bytes of body text over the inline memories that are in force: an expired inline memory, an `index` memory however long its body, and an empty body spend nothing. The band is rendered into the opening message once per session and again after each compaction, so what the budget admits costs prefix tokens the cache holds, not tokens on every turn.

A write that would take the tier past the budget is refused at every door alike: `memory_write`, `memory_edit`, `meka memory add`, `meka memory edit` and `PUT /v1/memory/{name}`. The refusal states the total the tier would have reached, names the largest inline memories as they would stand, and gives the remedy: shorten one, or set `load: index` on one that is not a standing rule. Nothing of a refused write lands. A write that shrinks the tier, or leaves its size alone, always goes through, so a store over the budget can be brought back within it one memory at a time.

```
inline memories would total 33.4 KiB against the 32.0 KiB inline budget (`[memory]
inline_max_bytes`); shorten 'handbook' (20.1 KiB), 'deploy' (6.2 KiB) or 'tone' (1.1 KiB), or
set `load: index` on one that is not a standing rule
```

The budget is what the agent reads whole, and the doors keep a store within it. A store can still be over it: the key lowered beneath what the store already holds, the upgrade that made every priority-0 memory `inline`, or a row written past meka's doors. The band then shows the bodies that fit, in index order, and lists the rest by description under a sentence stating the total against the budget and asking the agent to read one before relying on it and to shorten them or change their load; that list is held to the index's own ceilings and counts what it cannot list, with the same tag histogram and `memory_search` pointer the index footer carries. No body is ever cut partway: a rule is shown whole or named as not shown. Mid-session the context changes carry the same: a memory pushed out of the band is named, and one that fits again after a cleanup is restated whole, since the agent has only ever seen its description.

## The index budget

The index is capped at 8 KiB and 200 entries. When more memories exist than fit, the section ends with a line stating how many were left out, and, when they carry tags, what they are about:

```
4910 more memories not shown here, most common tags infra (820), people (611),
decisions (405); use `memory_search` to find them.
```

A bare count is not a usable signal once it runs to thousands: it says something is missing without saying what. The tag distribution is something the agent can turn into a query, which is most of what tags are for.

Nothing is lost. `memory_search` covers the whole store, including the entries the index omitted.

## Reading long notes

`memory_read` returns up to 16,000 body characters per call. When more remain, the result gives a
continuation offset. Pass it as `offset` with the same memory name to read the next part. Offsets
are zero-based character positions in the displayed body, not byte positions. The description and
age are repeated on each page.

## Search

`memory_search` is the primary way to reach a store larger than the index can show. It is backed by a SQLite FTS5 index over the same table.

**Ranking** combines three things, so the result is what you probably meant rather than merely what matched:

- **relevance**: BM25, weighting a hit on the name above the description, and the description above the body.
- **importance**: the declared priority, blended with how often you have actually read the memory. A memory opened forty times is important whatever it was labeled two years ago, which is the counterweight to priority drift.
- **freshness**: a gentle decay on `verified`, which is `created` until the agent verifies the fact, **disabled entirely for priority 0–1**. A two-year-old standing rule is exactly as binding as a new one; a two-year-old situational note probably is not, unless somebody checked it last week.

**Fuzzy matching** works in four senses, and the result says which one answered so a guess is not mistaken for a recalled fact:

| Kind | Example | How |
|---|---|---|
| Word endings | `preference` finds `prefers` | Porter stemmer, always on |
| Diacritics | `viet` finds `Việt` | Every diacritic is folded, always on |
| Typos and truncation | `Tokoy`, `Tok` | Retried as a prefix match, then by spelling distance |
| Word *beginnings* | `deployment` finds `deploy` | The prefix retry also works the other way |
| Unsegmented text | `深圳` inside `办公室在深圳南山区` | Matched as a literal substring, beside the word matches |
| Different wording | `verbosity` for `terse` | Pass several phrasings in `queries` |

The second and third rows are the two the stemmer alone does not cover. SQLite's Porter strips inflections (`deploys`, `shipping`, `running`) but not every derivation: `deployment` does not stem to `deploy`, so a search for it used to miss a memory whose body says `Deploys`. The prefix retry therefore runs in both directions, shortening the *query* as well as matching the start of the stored word, and says it was a prefix match either way.

The fourth row is why word-splitting is not the whole story. The tokenizer divides on non-alphanumerics, so Chinese, Japanese and Thai prose, and a long identifier, path or URL, arrive as a single token that only matches in full. A query term in a script written without spaces is therefore matched as plain text every time, beside the word matches for the rest of the query, and a memory holding both ranks first; the result names the terms the scan answered. When nothing else answers at all, the whole query is scanned as plain text, and the result says that is what it did.

What the matching does not do: an Arabic word is matched as written, article prefix included; a German compound is not split into its parts; Thai is matched by its consonants alone.

The last row is the important one: `queries` is a **list**, and supplying synonyms costs nothing. `["terse", "brevity", "verbosity"]` in one call finds a memory that used any of them, which is the answer to "the agent has to guess the words it used months ago": it does not have to guess right, only to guess several times.

Results carry enough to act on without a follow-up read: name, priority, age, read count, description, and the body itself when it is short, with "verified" when the fact was verified after it was recorded and "expired" when it has expired.

### Filters

`memory_search` takes three filters beside `queries`, and so do `meka memory list` and `GET /v1/memory`, so a hygiene pass reads the same rows whoever runs it:

| Filter | Selects |
|---|---|
| `referencing` | Memories whose references name this memory, path, URL or id, matched whole and without regard to case. |
| `unverified` | Memories not verified within this long: `90d`, `6months`. |
| `expired` | Memories whose expiry has passed. |

With `queries` a filter narrows the search. Without them it is a listing, in index order, most important first, and the result says which: "3 memories referencing 'deploy-host', most important first". A scheduled job can run one (`memory_search` with `unverified: "90d"`, then `memory_read` and `memory_write` with `verified: true` on what still holds) and keep the store honest without anyone remembering to.

## References and backlinks

A memory's `references` say what it depends on: other memories by name, a path, a URL, an external id, one line each and at most twenty. They are structured so the question "what else cites this?" has an exact answer:

- `memory_read` shows what a memory references, with the description of each reference that names a memory, and what references it ("Referenced by: ..."), so a changed fact can be revisited everywhere it is cited.
- `memory_search` with `referencing`, `meka memory list --referencing` and `GET /v1/memory?referencing=` list the memories pointing at a target.
- `memory_delete` names the memories that still reference the deleted one and leaves those references in place, as markers of a fact to revisit.

A reference that names no memory is allowed: it is a pointer to something worth writing later, and the read shows it bare where a resolved one carries a description.

## Source and verification

`source` is one line saying where a fact came from: who said it and when, the log it was read from, the page it was taken from. It is shown on `memory_read` and `meka memory get` and takes no part in ranking.

`verified` is when the fact was last verified to hold. It answers a question neither other timestamp can: `created` is when the note was first written, and `updated` moves on any change to the row, a retag included, so neither says whether anyone has checked the note since. The stamp is set at creation and moved only by an explicit `verified: true` on `memory_write` or `memory_edit`, never as a side effect of an edit, because a stamp that moved on every write would answer the same question `updated` does. A write that states newly learned information should pass `verified: true` with it.

What the stamp buys: `memory_read` captions a memory "Recorded 47 days ago, verified yesterday" or "Recorded 47 days ago, not verified since"; search freshness decays from it, so a re-verified old note ranks as fresh without a rewrite; and the `unverified` filter lists what has not been checked within a period.

## Expiry

`expires` is when a memory stops applying: an RFC 3339 timestamp, or a `YYYY-MM-DD` date meaning the whole of that day in local time, so a note that holds "until the freeze ends on the 25th" expires when the 26th begins. Past it the memory leaves the index and the inline band, stays searchable, and is marked as expired wherever it is read or found; nothing deletes it. The index counts its expired memories, and the context changes announce one expiring between turns, since an inline rule that expires has stopped applying without its text changing. Omitting `expires` on a write keeps the expiry, a rewrite of an expired memory keeps it expired and says so, and `""` clears it.

### The search index

The FTS index is an [external-content](https://sqlite.org/fts5.html#external_content_tables) table over `memories`, kept in step by three triggers. It is **derived and disposable** even though the memories themselves are not:

```bash
meka memory verify              # check the index
meka memory verify --rebuild    # regenerate it from the table
```

`verify` checks two things: that the index is structurally sound, and that it holds exactly as many documents as the store does. It deliberately does not claim more. FTS5's own `integrity-check` does **not** compare an external-content index against its content table, so a memory whose text changed while a trigger was not firing leaves both checks happy; only searching for the new wording reveals it. If search is missing something you know is there, rebuild; it is one pass over the table and cannot lose a memory, because the index is derived.

## Agent tools

| Tool | Purpose |
|---|---|
| `memory_write` | Save a memory, or update one by writing to the same name; `tier` picks the [tier](#tiers) |
| `memory_edit` | Replace an exact string in a memory's body |
| `memory_read` | Read a memory body, continuing with `offset` when needed |
| `memory_search` | Ranked full-text search over the memories the session sees, or a filtered listing; `tier` narrows it to one tier |
| `memory_delete` | Remove a memory permanently |

A write, an edit or a delete draws what changed as a diff on the screen and in an ACP editor, over the text `meka memory export` writes, so a reworded description or a moved expiry is one changed line of frontmatter and a rewritten body shows line by line; the HTTP feed carries the same two texts on `tool_call.completed`.

`memory_read` states when the memory was recorded and when it was last verified, and notes that it is a point-in-time observation. A memory recorded months ago is not live state, and an old note asserted as current fact is the failure this guards against. It is also the only thing that increments the read count: a search hit is weaker evidence, and an operator reading through the HTTP API is not the agent recalling anything.

A write whose description runs past 200 characters gets a note in the result and nothing else: the index lists the description every turn, so one that has grown into a paragraph is spending the budget of several entries, and the fact belongs there with the detail in the body.

`memory_write` also names an existing memory whose description says close to the same thing, when there is one:

```
Saved memory 'alice-tz' (priority 5). It is in your memory store from the next
turn on, and `memory_search` will find it whatever the index has room to list.

Note: 'alice-timezone' already says something very similar. If this is the same
fact, call memory_write on 'alice-timezone' instead and delete 'alice-tz'. Two
near-copies both stay in the index for ever and neither supersedes the other.
```

This never blocks the write. The failure worth preventing is the silent one, where a store grows a hundred near-copies because nothing ever mentioned the ninety-nine.

## What not to save

Memory is for what is *not* derivable from the material at hand. Code structure, git history, and file contents are all reachable with `file_search`, `file_read`, and `shell_execute`, so recording them produces stale duplicates of things the agent could just look up.

What belongs in memory: who someone is and how they prefer to work, guidance you have given that should not need repeating, decisions and their reasons, and pointers to where information lives in external systems.

## CLI

```bash
meka memory list                                    # index order
meka memory list --referencing deploy-host          # what cites a memory, path, URL or id
meka memory list --unverified 90d                   # not verified in 90 days
meka memory list --expired                          # past their expiry
meka memory list --project webdev                   # one project's tier; --project "" the global tier
meka memory get k4yt3x-prefers-terse-replies        # every stored field, backlinks included
meka memory show k4yt3x-prefers-terse-replies       # the body
meka memory list --format json                      # {"memories": [...]}; get and show print one object
meka memory add tz --description "K4YT3X is in UTC+8" --priority 2 --tag people
meka memory add tz --force --description "K4YT3X is in UTC+9"   # keeps body, tags, priority
meka memory add tz --force --description "K4YT3X is in UTC+9" --verified   # the fact was checked
meka memory add rule --description "Answer in kind" --load inline --body "Match the user's language."
meka memory add runbook --description "Where the deploy runbook lives" --body "wiki/ops/deploy" --load search
meka memory add freeze --description "No deploys this week" --expires 2026-10-25 --source "ops, in chat"
meka memory add notes --description "Meeting notes" --from-file notes.md   # the body, from a file
meka memory add stack --description "Uses pnpm" --project webdev   # filed in a project's tier
meka memory get notes --project movie                # the one in movie, when the name is in several
meka memory edit stale-note                         # $VISUAL, then $EDITOR, on the body
meka memory remove stale-note
meka memory export --directory ~/backup/memory      # one Markdown file per memory
meka memory import ~/backup/memory                  # the reverse, into this store
```

`meka memory add` takes:

| Flag | Purpose |
|------|---------|
| `--description <DESCRIPTION>` | Required. The one line shown in every session's memory index. |
| `--priority <PRIORITY>` | `0` is most important, `9` least; defaults to `5`. |
| `--tag <TAG>` | Label for grouping and filtering; repeatable. |
| `--body <BODY>` | Detail read on demand, or in context every turn with `--load inline`. |
| `--from-file <PATH>` | Read the body from this file instead of `--body`. |
| `--load <LOAD>` | `inline`, `index` (the default) or `search`. |
| `--reference <REFERENCE>` | A memory name, path, URL or id the memory depends on; repeatable. |
| `--source <SOURCE>` | Where the fact came from. |
| `--expires <WHEN>` | RFC 3339, or `YYYY-MM-DD` for the whole of that day; an empty value clears it. |
| `--verified` | Record that the fact was just verified current. |
| `--force` | Update an existing memory instead of refusing; whatever is not mentioned is kept. |
| `--project <NAME>` | The [tier](#tiers) to file it in, an existing project or `""` for the global tier; omitted, an existing memory keeps its tier and a new one is global. Naming the other tier for an existing memory moves it. |

On the screen, `/memory` is a picker of what is saved, by name and description, and Enter (or
`/memory <name>`) shows one memory in full.

`--format json` prints each memory with the fields [`GET /v1/memory`](http-api.md) uses (`name`, `project`, `description`, `priority`, `load`, `tags`, `references`, `source`, `created_at`, `updated_at`, `verified_at` and `expires_at` as RFC 3339) plus `read_count`; `get` adds `referenced_by`, `show` adds `body` as stored, and `list` leaves both out.

`meka memory edit` opens the **body** only. Metadata goes through `meka memory add <name> --force --description ...`, which keeps whatever it does not mention.

## Export and import

`meka memory export` writes one `<name>.md` per memory: YAML frontmatter carrying `description`, `project` for a project memory, `priority` and `load` when they are not the defaults, `created`, `verified` when it differs from `created`, `expires`, `source`, `read_count`, `references` and `tags` when they are set, followed by the body. A project's memories go under a directory of the project's name, so two projects holding one name write two files; the global tier is at the root. Beside the files sits `export.toml`, carrying the snapshot's `format_version`. `--project <name>` exports one tier. That is the `grep`, git and backup answer now that memories live in the store rather than in files.

`read_count` is there because it is the one value a file cannot otherwise reconstruct. Descriptions, bodies and dates are all in the note; how often the agent has actually opened it is not, and a restored backup with every counter at zero would silently lose each memory's accumulated ranking weight.

```bash
meka memory export --directory ~/notes/memory  # must be new or empty
```

The directory must be new or empty. An export is a snapshot, and merging into an existing one would leave a stale file behind for every memory deleted since, so it would never quite match the store. An export that fails partway removes what it had written rather than leaving a truncated snapshot, which would otherwise restore as a plausible fraction of your store.

The export directory is created at mode `0700` and each file at `0600`, and an existing empty directory is tightened to `0700`. A memory body is a private note and the store it came from is `0600`; publishing the same text world-readable because that is what the umask said would be a strange way to take a backup.

What lands on disk is byte-exact: bodies, tags, priorities and recorded dates are written exactly as stored, including zero-width joiners, CRLF line endings and leading or trailing blank lines. `read_count` rides along too, because it is the one value the rest of the file cannot reconstruct.

Descriptions are the one field normalized rather than preserved: every write door collapses a description to a single line before storing it, so what comes back is what was stored. A description made only of characters YAML cannot carry has no such form, and `meka memory export` refuses the whole run and names it rather than writing a file whose frontmatter would not parse.

An export reads back with any tool that understands YAML frontmatter, and `meka memory import <dir>` reads it back into a store: every file of the snapshot in one transaction, so nothing lands unless everything does. The project of each memory is read from its frontmatter and must exist, so `meka project add <name>` comes first on the other installation; `created`, `verified` and `read_count` are restored as written, which is what makes the import a restore rather than a copy. A memory the store already holds in the same tier is refused by name until `--force`, which overwrites it with the file; a name a global memory and a project memory would then share is refused outright, since no tier may hold the other's name; and the inline budget holds over the whole import as it holds over a write. The manifest is read first: a snapshot an older meka wrote at another `format_version` is brought forward in one place, the migration module, or refused by version.

The store stays the one source of truth. An export is a snapshot nothing reads until you import it, and an import is a one-time act, the way a session archive's is; meka never watches the directory.

## Coming from a file-backed store

Memories used to be Markdown files in `<config>/memory/`. If you are upgrading from 0.41, the one-shot migration script attached to the 0.42 release imports them into the store; run it once, check `meka memory list`, then remove the directory yourself. meka never reads those files again. What it brings forward on its own is the store; a directory of files you still have is yours to import when you get to it, and importing it twice is not something a startup pass could ask you about.

## Configuration

Memory is on by default. To turn it off:

```toml
[memory]
enabled = false
```

Disabling it keeps the five `memory_*` tool schemas out of every request and renders no memory section, which is worth doing if you run lean sessions that will never use it. Memories already stored are left alone, and both the `meka memory` subcommands and the `/v1/memory` endpoints still reach them: whether an *agent* keeps memories is a different question from whether you can inspect or back up what is already there.

There is deliberately no environment variable and no CLI flag: whether an agent keeps memories is a property of the installation, not something to vary per run.
