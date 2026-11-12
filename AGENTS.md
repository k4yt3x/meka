# AGENTS.md

Rules for changing Meka, for every agent and person working in this repository: general principles,
then Rust practice, then Meka's own design rules.

## What belongs in this file

Rules for changing this code that hold for anyone, on any machine, until the code changes, written
in the present tense with no dates and no incident stories. An entry earns its place when an agent
on a fresh clone would make a wrong change without it.

- How the code works belongs in `docs/book/src/` (`internals.md` for the internals), and what
  changed in `CHANGELOG.md` and the commit message.
- Facts about one machine or installation (hosts, toolchains, local paths, accounts) stay out of
  this public repository.
- A rule about the code settled in conversation lands here in the same change as the code it
  governs. An agent's own memory keeps at most the evidence behind a rule here and names its
  section, never a second copy of the rule.

---

# General principles

## The better design over compatibility

Meka does not keep a stable interface. When the cleaner shape breaks a config key, a flag, a tool
name, a route, a wire value or a stored shape, take the cleaner shape: mark it `**Breaking:**` in
the changelog, give the remedy in the upgrade guide, and convert stored data once in the migration
module. Never keep a worse design to spare users an upgrade step. Where a published standard (an
RFC, a protocol's spec) or a rule in this file settles a design choice, it decides, and a breaking
change it requires is made, not held back.

Backwards compatibility spread across readers costs one shim per reader per superseded shape,
forever. **Convert once, in one place**, so every other reader may assume the current shape
unconditionally. That assumption is the entire return, and it is lost the moment a second place
tolerates an old shape.

## Enumerate the doors

Most defects that survive review are one rule enforced at one entry point and not at its siblings.

Before writing a guard, list every path that reaches the thing being guarded (create, copy, fork,
import, re-root, resume, re-attach, patch, delete) and decide for each. Place the check where those
paths converge. A rule placed at a door is a rule the next door forgets.

**Prefer one definition over many checks.** When an invariant is asked in more than one place, give
it a single named predicate and call that everywhere. Consolidation beats a test per site, because
tests only confirm the sites you thought of.

**A guard sits ahead of every side effect it protects**, not merely ahead of the failure. Refusing
after a write leaves the write behind.

## Whose fact is it

If another system is the authority for a fact, ask it or let the user state it; never encode it. A
hardcoded fact about an external system expires, and nothing in the build notices.

- **The provider or service owns it**: a request parameter it defaults sensibly. Omit it unless the
  user asked for a value; omitting *is* how you request its default.
- **The user owns it**: anything neither side can determine, or where a wrong guess is invisible.
  One config key, one documented default; don't infer, probe or cache. State that default in the
  docs and, where a setup flow exists, on screen.
- **Meka owns it**: its own names, schema, and defaults for its own behavior. Encoding these is
  fine.

A guess about a request field Meka adds is tolerable when its wrong answer is a *rejected request*
and it fails toward omission. A guess that fails toward *sending* survives only where the endpoint
cannot vary, so never introduce one on a backend reachable through a user-supplied `base_url`. What
the conversation already holds is no guess: it goes back as the context holds it.

A value verified against a captured wire is a fact about the protocol rather than a prediction: pin
it deliberately and cite the capture. A capture pins a fact about the endpoint captured, so on a
backend whose endpoint varies, the captured value is the default of a user-owned key, never a bare
pin (`interleaved_thinking` on an `anthropic-messages` account is the pattern).

## Code

- Correctness and clarity first. Speed and efficiency are secondary unless stated otherwise.
- Comments explain *why*, never *what*: a constraint from outside this file, or why an
  obvious-looking alternative is wrong. Not history, not user-facing documentation, not argument for
  the choice; `git log` and `docs/book/src/` hold those. If it restates the code, delete it.
- Rationale is written in the present tense, as why the alternative is wrong now ("unbounded, a peer
  that never answers holds the turn"), never as the story of the defect that was fixed ("it used to
  connect with no bound"), and never names a past fix. No comment, test doc or warning carries
  knowledge of an earlier version; only the changelog, the upgrade guide and the migration module
  speak of the past.
- **Read the code, not the comment.** A comment is a claim about the code, not evidence for it, and
  where the two disagree the comment is usually what was updated last and least. Verify a stated
  invariant against the implementation before relying on it, and correct the comment when it is
  wrong.
- Add functionality to existing files unless it is genuinely a new component; avoid many small
  files.
- No creative additions beyond what was asked.
- Full words in names, no abbreviations.

## Verification

Verification is graded by what it catches, not by how much of it there is.

- **Per change**: build, the test suite, the exact CI gate (see "Build gate"), and a check of the
  change against its doors. Then **fake-guard every test written to protect a fix**: neuter the fix,
  confirm the test fails, restore it, confirm it passes. A test that cannot fail is worse than no
  test, because it reads as coverage.
  - Restore by reversing the exact edit: a version-control checkout also reverts every other edit in
    the file, and a backup moved back keeps its old mtime, so cargo reuses the neutered build. An
    integration test rebuilds `target/debug/meka` from the neutered source, so build again after the
    last guard before running the binary by hand.
- **Per release**: one structural review, docs and changelog. Don't run a cross-platform suite by
  hand per change; CI runs the matrix on every push.
- **Rarely**: mutation testing, when a subsystem is new, never as a gate; its yield falls as the
  suite densifies while its cost grows with the codebase.

Reproduce a defect before fixing it, and re-run the reproduction after. A fix verified only by a
passing suite was verified against the thing that already missed it.

Don't grow the suite reflexively: a test earns its place by being able to fail.

**A test never touches the repository or the developer's installation.** It never runs `git`, never
calls `set_current_dir`, and writes only under a temporary directory, because the working tree is
someone's uncommitted work. A test that runs a shell or a file tool works in `tempfile::tempdir()`
or the harness's `work_dir()`; `workspace::cwd_for_test()` is for a tool that only reads. A store in
a test is `Store::for_test()`: `Store::open(None, ..)` is the developer's own `meka.db`, and
`only_main_opens_the_store_at_its_default_path` in `tests/layering.rs` refuses it outside
`src/main.rs`.

## Changelog and documentation

- Update the mdBook docs under `docs/book/src/` for any user-facing change, and the upgrade guide
  for anything marked `**Breaking:**`.
- Update `CHANGELOG.md` for every change a user can notice, under `[Unreleased]`.
- [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/): only Added, Changed, Deprecated,
  Removed, Fixed and Security, in that order, grouped by type.
- `Fixed` = the behavior was wrong. `Changed` = it worked as intended and now works differently.
  `Security` = a vulnerability closed or an advisory answered, led by its CVE or RUSTSEC id;
  hardening that closes no known hole is `Changed`.
- The changelog is written for end users and integrators: what they will see, break on, or must do,
  in plain present tense. No internal names, module paths, test names or contributor notes; those
  belong in `docs/book/src/internals.md` and the commit message.
- One line per entry, under 100 characters. A change that needs more is two entries, or the docs
  hold the detail and the entry names the page. Trivial changes are compacted into one line or left
  out.
- Breaking changes get an inline `**Breaking:**` prefix inside their type, not a separate section.

## Prose style

- American spelling everywhere Meka owns the word: prose, comments, docs, strings, identifiers and
  Meka's own wire values (`canceled`, `color`, `catalog`, `behavior`). Only a name another protocol
  or crate defines keeps its spelling (ACP's `stopReason: "cancelled"`, MCP's
  `notifications/cancelled`, `serde::Serialize`), and a value a model emits may be accepted in both
  spellings while Meka writes one.
- No em dashes (`—`). Prefer a colon, a comma or parentheses.
- Sentence-case headings in the book; product names keep their case.

---

# Rust practice

## Safety

- Avoid panicking calls (`unwrap()`, `expect()`, unchecked indexing). Propagate with `?`. The lints
  are `warn` in `Cargo.toml` and relaxed under `cfg(test)`, where panicking on failure is the point.
- Never discard errors with `let _ =`. Propagate with `?`, log explicitly when ignoring is correct,
  or handle with `match` / `if let Err(..)`.
- Errors from fallible async work must reach the UI layer so the user gets real feedback.

## Layout and style

- No `mod.rs`. Use `src/some_module.rs`.
- New crates set `[lib] path = "..."` in `Cargo.toml` for a descriptive root name.
- Never hand-wrap comments: one line per paragraph, which `cargo +nightly fmt` wraps
  (`.rustfmt.toml` sets `wrap_comments = true`). If a wrap lands awkwardly, reword rather than
  inserting a manual break; "Build gate" covers the comments rustfmt declines to wrap.
- In async contexts, scope a clone by shadowing the binding in a block around the future:
  `spawn({ let state = state.clone(); async move { … } })`.

## Names and visibility

- Test names read as sentences: `a_fork_keeps_its_images_when_the_source_is_deleted`, never a
  `test_` prefix. A test-only constructor is `for_test()` or ends in `_for_test`.
- `new` is infallible; `open`, `from_*` and `resolve` are fallible. No `fresh`, `try_new` or
  `from_connection`. Predicates are `is_*`; the feature-on question is `is_enabled()`.
- `pub(crate)` for anything another module reads, `pub(super)` for a parent alone, never bare `pub`
  in this binary crate; child modules are `pub(crate) mod`.
- Every public item has a `///` comment, except an axum handler (its route documents it) and a
  tool's struct (its `definition()` does).
- Time is a `Duration` constant; `_MILLIS`/`_SECONDS` only for an integer that goes on a wire. Sizes
  are `_BYTES` or `_CHARS`, never `_LEN`, `_LIMIT` or `_CAP`; a MiB literal goes through the named
  `MIB`. Terminal display goes through `text::format_timestamp` and `text::format_size`; the wire
  carries RFC 3339 and raw byte counts.
- A value enum with a wire spelling follows `Backend`: one `const fn name()`, `Display` and
  `FromStr` derived from an `ALL` table plus `name()`, serde through `try_from = "String"` and
  `into = "String"`, clap parsers `.parse()`. One spelling per value; no aliases, no alternates.
  Values a model emits (checklist status words) are the one exception.

## Build gate

Run `cargo +nightly fmt` and `cargo sort -w` after editing. Before declaring done, reproduce CI's
exact gate:

```
cargo +nightly fmt --check
cargo sort -w --check
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items
cargo test --locked                 # CI adds --features mock-provider; debug builds carry it anyway
cargo check --locked --all-targets  # on the MSRV in Cargo.toml's rust-version
mdbook build docs/book
```

CI denies warnings on clippy and rustdoc, so the bare commands can pass locally and fail CI. It runs
them on the current stable, which may be newer than yours and is where a deprecation lands first:
when yours is older, run the gate on the stable `gh run view` reports. It formats with the current
nightly rustfmt, whose layout moves between nightlies: run `rustup update nightly` before
`fmt --check` when yours is more than a few weeks old.

- `--all-targets` matters: plain clippy skips tests and benches.
- The `mock-provider` feature exists so a release-profile build can run the suite; debug builds
  carry it regardless. A shipped artifact must never have it, because `MEKA_MOCK_PROVIDER=1` then
  stands in for every profile.
- In rustdoc, watch `rustdoc::invalid_html_tags`: a bare `<word>` parses as an unclosed tag.
  Backtick it, or rephrase if the comment is also a clap help string, where backticks render
  literally.
- `fmt --check` does not enforce `comment_width`: `wrap_comments` silently declines some comments
  (in a macro body, in a method chain) and still exits 0, so a paragraph left for rustfmt to wrap
  can ship at 400 columns. After `fmt`, run `awk 'length>100 && /^[[:space:]]*\/\//'` over the
  changed files and reword what it prints, or break it by hand.

## Clap help text

`///` doc comments must render within 120 columns under `-h` and stay as short as they can: no
examples, no tautology, one line where one line says it. Verify by running the binary with
`COLUMNS=120` for every changed subcommand: source length ignores clap's indent, value-name width
and auto-appended hints, and a new flag widens the whole column, so it can push existing lines over.

A `-h` summary is one clause that says what the thing is, without restating the flag's own name. It
carries no tail: no cross-reference to a config key or another command, no constraint clap enforces,
no consequence, no backend-only qualifier; that detail is one `--help` or one book page away. What
it keeps:

- a value list, built through `listed(...)` from the enum's `ALL` table so it cannot drift
  (`--format` through `output_format_help()`);
- `(repeatable)`, and a `(default: …)` that tells the reader what omission means.

A group's summary may be plain ("Manage profiles") unless a fact its name lacks earns more ("Manage
accounts and their credentials"). Don't reword a line only to avoid ending on a preposition
("Profile to use for this session" stands). One verb per shape: every `show` subcommand says "Show",
the `set` commands read alike, and a rename's first positional is "Current name".

Order: positionals, then the required option, then options grouped by what they govern, `--format`
last; a command that writes config follows its config struct's field order. The run's flags go input
(`-p`, `--skill`), session (`-c`, `-r`), bindings (`--profile`, `--project`, `--permission`,
`--instructions`), confinement, tools, output, verbosity.

Long-form prose goes after a blank `///` line so it appears only under `--help`; when that prose is
multi-line or indented, add `#[command(verbatim_doc_comment)]`. Command summaries take no trailing
period; multi-sentence prose is punctuated normally.

---

# Meka

## What Meka is

Meka is a general-purpose agent harness, not a coding agent. Nothing in it assumes the working
directory is a source repository: no `.git` special case, no checkout as the unit of work, no
knowledge of build tools or version control in its behavior or in its own prompt text (the
compaction summary asks for rules, commitments, facts and the next action, never files, code or
errors). Repository conveniences belong in user-authored skills and instructions. A mechanism
borrowed from a coding agent keeps the mechanism and leaves the repository premise behind; a feature
that needs to know what the workspace holds takes a config key the user sets, or is dropped.

Meka is designed as the general product, never for one client. A downstream client's proposal is a
problem statement, not a shape: derive the general primitive, give each independent choice its own
field with no refused combination, put each fact at the level it belongs to (a session's on the
session, not on each request), walk every existing door against it, and judge its defaults by a
plain `curl`, not by the client that proposed it.

## Meka's own content is typed

What Meka writes into a conversation, or anywhere a reader will later need to tell it from someone
else's words, is a block type or a field, rendered to text only at a provider or client boundary.
Nothing outside the migration module reads meaning back out of text Meka wrote: no prefix check, no
comparison with a placeholder, no pattern over a tag. A label in rendered text (`[Meka harness]`, an
inbox header, a summary's headings) is formatting, spelled only by the function that renders it;
`tests/layering.rs` holds each to its renderer through its `LABELS` ledger. A new label is a new
ledger entry, never a new reader.

## What a model returns is recorded as returned

What a provider returns is recorded as it returned it, part for part, with how it stopped, the
model it names, the profile that sent the request and what it spent. Each object it returns maps
to the element that stands for it (the response to the reply's record, a message to its `Message`,
a part to its block), which keeps what nothing else holds as `provider_fields`; an object with no
element is `Unrecognized`. A stream is folded as the protocol's reference client folds it, never
by a rule of Meka's, and what restates Meka's own request is not the response's. Nothing Meka writes
is recorded as the model's: a reply with nothing in it is stored with nothing in it. A rewrite of
the context (a repair, a redaction, an unsign, a compaction's kept tail) never changes the record;
it is a row of its own over the view. What a surface shows of a reply, how it joins parts or words
an empty one, is decided where it is drawn. What goes back on each wire is the next section's.

Meka implements each protocol as its spec defines. An endpoint that claims a protocol and deviates
from it is not accommodated, and a shape the spec does not allow is recorded verbatim, not parsed.
A vendor's documented extension is supported as users need it.

## What goes back to a provider

Meka is built for users who know what they are doing. It follows each protocol's spec exactly and
carries as much of the conversation as each protocol has room for. What it does when a provider
rejects part of that turns on who chose the thing that broke: a person's choice is theirs to
handle, and what nobody chose, Meka repairs where the provider sees it and says so.

1. **Faithful mapping.** Every request sends the conversation the context holds, whichever
   provider, host or model produced each part, and each part goes into the slot its protocol's
   spec, or a vendor's documented extension of it Meka supports, defines for that kind of part.
   What the current protocol has no slot for stays in the record, and a switch back sends it
   again, unless a repair took it out of the view.
   - Text, tool calls and tool results have a slot on every protocol, and images wherever the
     protocol carries them.
   - Readable reasoning goes into the protocol's slot for reasoning: DeepSeek's
     `reasoning_content` on Chat Completions, a `reasoning` item on Responses. A part keeps its
     kind: a summary (a Responses summary, Claude's thinking text when the request asked for it
     summarized) goes as one, and raw reasoning (DeepSeek's, a Responses reasoning text, Claude's
     otherwise) as raw. Claude's protocol requires a signature on every thinking block, so it has
     no slot for reasoning that came without one.
   - Opaque reasoning goes only into a slot of the protocol that issued it. A Responses blob goes
     in a `reasoning` item's `encrypted_content`, with the id it came with, to the host that
     issued it, under any model there: only its issuer can open it, and as far as Meka can see
     the host is the issuer. A Claude signature goes on its thinking block, and a redacted block
     as itself, to any Claude endpoint; one that cannot verify them rejects them, and rule 4's
     retry takes them out.
2. **No guessing for the provider.** Beyond rule 1's binding of a blob to its host and rule 4's
   repairs, Meka never drops or alters what the conversation holds because a provider might
   reject it, and binds no part to a model or an account. Whether a provider accepts a part is the
   provider's fact; one that rejects what it returned itself, sent back as the spec allows,
   deviates from its protocol, and its rejection is reported.
3. **What a person chose, they handle.** A switch of the session's own profile, or of that profile's
   model, is a person's choice, and so is the switch of every sub-agent that follows its parent's
   profile. A change of model is seen only by the agent that had a request accepted on the old one:
   a stored reply records the model its response named, which need not spell the configured one, so
   a fresh process, an HTTP re-attach or an ACP load compares profile names alone, and a rejection
   after a model change there takes rule 4's path. When the provider rejects the first request after
   a switch (a 400 or 422; an outage is no verdict and takes rule 4's path), the turn fails with the
   provider's own error, nothing of it is degraded, and Meka adds how to go on: switch back,
   compact, switch again. A sub-agent adds nothing, since its failure reaches its parent. A
   compaction a person asks for keeps nothing verbatim (only `POST …/compact` can ask for a tail),
   so the new provider receives the summary and nothing the old one returned; compacting after the
   switch sends the same history and fails the same way. A fallback is a switch the operator
   configured, so the operator tests that the profiles in a chain survive switching between them
   before relying on it: a rejection while a fallback serves is not degraded, and the session's own
   profile coming back after one is no switch.
4. **What nobody chose, Meka repairs in the view, and says so.** When what breaks a request is
   nobody's choice (a tool's output, a provider's own odd reply, Meka's own edit), Meka changes
   what the provider sees so the session goes on, and leaves the record whole. A repair the
   session keeps is announced with a warning or by the event that caused it; a repair to a
   compaction's own request, a copy thrown away after, is only logged. An unattended agent must
   not stall on something no one could have prevented. The cases:
   - **Degrade and retry.** Content added since the last request a provider accepted, which no
     provider has accepted yet (a tool's result, an attachment), is degraded when a provider
     rejects it: attachments first, then the tool exchanges, each tried once and undone if it
     does not help. Without it, an agent that read one file a provider will not take could not go
     on without someone editing its history by hand. After a compaction the window is all the
     summary did not replace; after a switch, rule 3 applies instead.
   - **Claude's signature retry.** Claude binds a signature to the conversation before it, which
     Meka's own compaction or repair changes, so a rejected signature is retried without the
     conversation's thinking, recorded as an `Unsign` repair. The rejection does not say whether
     an edit or a switch caused it, so the retry takes out both.
   - **Dangling reasoning on Responses.** A reasoning item with no output after it is not sent,
     since OpenAI rejects one, and Meka's cut replies and nudges and a reply that stopped
     mid-reasoning all leave one. An endpoint that returns reasoning after its output does not get
     that reasoning back.
   - **An empty assistant message on Claude.** One left with nothing Claude's protocol carries
     goes as `[No message content]`, since Claude rejects an empty one.
5. **The record is never healed.** It keeps what was returned; a repair is a row over the view.

## Vocabulary

One word per concept, everywhere it is written or read:

- **Meka** wherever the product is named: prose, comments, every message, log and help line, and
  a display-name field (an ACP or MCP handshake's `title`, the Windows product name and file
  description, the name `--version` prints). `meka` is only an identifier: the command and a
  command line, the binary's file name, the crate, paths and `meka.db`, a tracing target, and an
  identifier on a wire (`meka-priority`, the `meka/<version>` user agent, a handshake's `name`).
- **permission level**, never "mode": `level` in prose, `permission` in identifiers. ACP's own
  `set_mode` and `mode_id` keep their protocol names.
- **profile**, **account**, **backend**. "Provider" is only the upstream service and the `Provider`
  trait: "provider temporarily unavailable" is right, "provider profile" is not.
- **store** for the `Store`: `store` in identifiers, "the store" in prose, `meka.db` for the file.
  `manager` is only `McpClientManager`.
- A session's **conversation** is told two ways. Its **history** is what was said: every turn, the
  ones a compaction summarized included, each once where it was said, without what a rewind took
  back. Its **context** is what the model sees now: the latest summary, the turns it kept, and what
  came after, which compaction, repair, image redaction and rewind rewrite. The **event log** is
  the rows on disk, said only where storage is the point; a row's **position** is its place there.
- A **turn** is a row of `turns`, the one meaning the HTTP API has; what a person typed is a
  prompt.
- **recall** for the lines typed on the screen that Up and Ctrl+R bring back, never "history"; one
  is a recall entry.
- **sub-agent** as the noun; "worker" only when the orchestration role is the point; never
  "delegate" as a noun. Depth 0 is the **root**, the spawner is the **parent**.
- **scratchpad entry** in every string and identifier, the table `scratchpad_entries` included.
- **approvals** is the switch; the prompt is an **approval prompt** with the header `[approval]`;
  its answers are allow and deny.
- **standing instructions** for the concept; "instructions file" only for the file on disk.
- **id** lowercase; "UUID" only where the format is the point.
- Refusal verbs: *refuse* is Meka's own decision; *reject* is a remote party (a provider, an MCP
  server); *deny* is the OS, a permission, or the user's answer; *decline* is a gate or the model.

## Commands, messages and listings

- **Command names.** A top-level command whose verbs act on instances names the object in the
  singular (`account`, `profile`, `session`, `skill`, `memory`, `schedule`, `tool`, the screen's
  `/task`); a command over one thing keeps the concept's own word (`recall`, `instructions`).
  `mcp tools <name>` lists a server's tools and is a verb, not a noun.
- **`--format`** is the one output-format flag, on the run and on every `list`, `show` and `export`:
  `plain` by default, `json` where a command offers it, value sets documented per command. Never a
  `--json` switch.
- **What a message says.** A message the user reads (error, warning, notice, hint, help) says what
  happened and, when there is exactly one, the remedy: no examples, no alternatives, no explanation
  of internals, no second sentence that restates the first. One sentence where one suffices. A
  printed line stays within 120 columns where its content allows, so it fits a terminal without
  wrapping.
- **Capitals and periods.** Every line Meka says in its own voice (a status line, a notice, a stage
  direction, a turn's end row, a note printed on the way out) starts with a capital letter whatever
  its shape ("Background task 7f3a1c22 finished after 6s", "Interrupted"); a line that opens with a
  number or an acronym keeps that. One that is a full sentence ends in a period ("Connected to
  'exa'."); one that is a fragment, or ends in a value, id, path or time, does not ("Profile set to:
  work", "Finished at 14:43 after 1m 13s"). On the interactive screen none ends in a period,
  whatever its shape: the screen's own strings are written without one, and a row drawn from a
  sentence another surface shares (a notice, a stage direction, an error, a compaction's headline)
  loses it through `draw::without_closing_period`. A `-v` log row is not said in Meka's voice and is
  drawn as written.
- **An empty list** says "No <nouns>." on stderr.
- **One wording per event**, live and on a replay of the log, so a resumed session reads as the live
  one did. The live screen may say more than the log records, news as it happens such as a checklist
  nudge's counts or how late a scheduled fire ran; the replay leaves that out and says the rest in
  the same words, and a reload lets the live row go for the replay's.
- **Quoting.** Backticks for keys, commands and code (`` `[permissions].default` ``,
  `` `meka profile use` ``); single quotes for values and names (`'work'`).
- **Refusals.** An unknown name is reported through `text::unknown_name`, never a per-door sentence.
  A refusal whose remedy is a command is a typed `MekaError` variant (`ProfileDefaultStillNeeded`,
  `McpCredentialInTheWay`): its `Display` is the shell's sentence, and `ProblemDetail::for_error`,
  the one place every HTTP route takes it from, words the route from its fields.
- **Listings.** A listing is a `text::format_table` over a column declaration, never hand-rolled
  widths: an identifier the reader retypes is shown in full, text shown for information is capped (a
  name at `NAME_WIDTH`), and one prose column takes what is left of `TABLE_WIDTH`, never below its
  floor. Rows reach the printer raw; it sanitizes and cuts. A table handed to the model uses
  `format_columns`.

## Output: prints vs. tracing

**If the user doesn't have to see it to use the command, it is a log.** The default level is `warn`,
so `info!` and `debug!` are silent unless the user passes `-v`, `-vv` or `RUST_LOG`. Aim for quiet
on success: an `ok:` confirmation is a log, not a print, and the exit code carries success.

`println!` and `eprintln!` are only for requested data; content the user must copy, type or visit;
the lines around the screen (the session id on leaving); and hard errors. Everything else is
`tracing`. Don't invert it either: a command's primary output is never a `tracing::info!`, or the
user needs `-v` to see what they asked for, and a config flag that asks for visible output is
honored, not demoted to `info!`. Drop preambles before the actionable line.

The stream is a contract:

- **stdout**: only the data the command was invoked to obtain. In a one-shot run that is the
  model's answer and nothing else; the interactive screen owns the terminal, draws everything on
  stdout, and refuses a pipe.
- **stderr**: everything else, including prompts, live UI, indicators, hints, status and errors,
  and every spacing blank line emitted around them.

Litmus test: `meka ... 2>/dev/null | next-tool` must leave only the requested data on stdout.

Levels: `error!` for an unrecoverable failure about to propagate or a turn whose outcome is lost;
`warn!` for a recoverable fallback, a rollback, or a lost write the user can act on (never `debug!`
for one); `info!` for lifecycle signposts; `debug!` for module-level diagnostics. Every log string
uses inline captures (`{name}`), and a failure reads "failed to <verb>", never "could not".

## Configuration surfaces

- **`config.toml` is the complete source of truth** for non-secret settings; every persistent
  setting lives there.
- **Accounts and profiles are config-only, never env**: an ambient variable must never silently
  rebind which account a named profile bills. An account (`[accounts.<name>]`) is a backend, an
  endpoint and the credential a login produced; a profile (`[profiles.<name>]`) is an account plus a
  model and every model-tied setting.
  - A session runs on the profile its own row names, and the row moves only by an explicit act
    (`--profile` on a resume, `/profile`, `PATCH /v1/sessions/{id}`, ACP
    `session/set_config_option`). A *new* session records `--profile`, else `default_profile`, else
    the sole profile. A sub-agent's row records the profile its `agent_spawn` call named, else its
    parent's live profile, which it keeps following.
  - `meka account` (`add`/`login`/`list`/`rename`/`remove`) manages accounts and `meka profile`
    (`add`/`set`/`use`/`list`/`rename`/`remove`) profiles, mirroring `meka mcp`. `profile use`
    points `default_profile` at a profile. `profile remove` refuses the default while two or more
    profiles would remain, and drops the key when one would; `profile add` names the profile that
    was alone when a second joins it, so the default stays put. `account login` rotates a credential
    without touching the account or its profiles, and `account remove` refuses while a profile names
    the account. A `rename` moves the name everywhere it is recorded (the credential row and every
    profile's `account`; `default_profile`, every session row and each pinned sub-agent spec) and
    changes no setting.
- **Secrets live in the database**: `account_credentials` keyed by account name and
  `mcp_credentials` keyed by `(server_name, kind)`, so two accounts, or a client secret and its
  refreshable bundle, can coexist. Every secret is read from stdin, never taken as an argument,
  because arguments are visible in `ps` and shell history.
  - A field that may *contain* a secret is not itself one; it stays in `config.toml` with `${VAR}`
    expansion.
  - Retiring a config key that held a secret gets no compatibility shim. It stops being modeled and
    `deny_unknown_fields` names the key and line; the upgrade guide carries the remedy.
- **Environment variables are operational only**: `MEKA_CONFIG_DIR`, `MEKA_DATA_DIR`, permission,
  instructions, sandbox backend, MCP tool timeout, `RUST_LOG`. The command line wins over the
  environment, which wins over the file, written as `cli.x.or_else(env).or(file)` in
  `ResolvedConfig::resolve`, and every layer below the hosts takes the result as a value.
  `tests/layering.rs` keeps every other file from reading a `MEKA_*` variable
  (`ENVIRONMENT_READERS`); `MEKA_CONFIG_DIR` and `MEKA_DATA_DIR` are `paths.rs`'s, because they
  locate the file `resolve` reads.
- **Session and display tuning is config-only.** No env vars or flags for set-once preferences.
- **Config sections keep one order**, the same on the reference page (`config-file.md`) and in
  `ConfigFile`'s fields, which is the order a file is best written in: the multi-entry blocks first
  (`default_profile`, `[accounts.*]`, `[profiles.*]`, `[mcp]`), then each flat table from the most
  to the least consequential (`[permissions]`, `[shell]`, `[tools]`, `[subagents]`,
  `[instructions]`, `[skills]`, `[memory]`, `[schedule]`, `[background]`, `[session]`, `[thinking]`,
  `[web]`, `[display]`), and `[serve]` last. A new section is placed by that rule in both, never
  appended, and a note that is not a key goes in the page's intro, never between sections.

## A profile is indivisible

A profile is a named bundle: the account it bills, the model, and every model-tied knob; the account
is the backend, the endpoint, the OAuth settings and the credential. A session selects a profile by
name and records that name. **Nothing overrides a field inside either**, or the run gets a
combination nobody configured and no field states the mismatch.

- **`--profile <name>` selects**, and is the only profile flag on a run. `-p` is the prompt.
- **`profile add` / `set` write profile fields; `account add` writes account fields.** `set` edits
  one key in place via `toml_edit`, preserving comments and order. It has no session scope. There is
  no `account set`: an account's settings are what a login was made against.
- **A field belongs on the profile when it is user-owned and model-tracking**, and on the account
  when two models on one endpoint would state it the same, per "Whose fact is it". A profile field
  gets a `profile add` flag and a `profile set` key, never a global CLI flag, env var, or session
  column. `account` is not settable on a profile, because moving it moves every session on the
  profile onto another credential; `device_id` is not settable at all, because Meka resolves it.
- **`backend` names the driver**, not a protocol: API-key backends are named for the protocol
  because `base_url` decides the endpoint, subscription backends for the product because the
  endpoint is fixed.

## Typed keys and `parameters`

A profile shapes the request two ways. A **typed key** is Meka's: a name Meka defines and writes in
each wire's own form. **`[profiles.<name>.parameters]`** is the provider's: fields Meka copies
into the request body without reading them, spelled as that provider documents them. A setting is a
typed key only when it answers yes to one of three questions:

1. **Is it Meka's own?** Meka acts on the value (computes from it, decides by it, or changes how it
   builds, reads or replays the conversation), or sends the field unasked for a feature of its own,
   where the key is the switch an endpoint that rejects the field needs. `thinking`,
   `max_output_tokens` on a Messages wire, `encrypted_reasoning`, `reasoning_summary` and
   `deferred_tools` are Meka's own. Typed, on the backends where that holds.
2. **Does every backend have it?** One concept every wire states, under whatever names: `effort`,
   `max_output_tokens`. Typed, as one name Meka translates.
3. **Does a subscription's official client offer it?** A subscription's request copies an official
   client's and takes no `parameters`, so a setting that client offers its user is a typed key on
   that backend alone, sent as the client sends it: `thinking_display` is Claude Code's display
   modes, `cyber_access_program` is Codex's Daybreak. An API-key backend copies no client's
   request, so there the same setting answers the first two questions like any other:
   `access_programs` is a parameter on `openai-responses`.

Anything else is a parameter: `temperature`, `top_p`, `stop`, verbosity, a server's own extensions.
Showing a value on `/status` is not acting on it, and translating a name is no reason on its own: a
concept only some backends share is written in each one's spelling, or the typed set becomes a
translation table that grows with every provider.

Each top-level field of a request has one owner, so nothing overrides anything. Meka reserves every
field it writes on a backend, whether or not a given profile makes it write that field, and a
parameter naming one is refused at load and at every write door through one predicate, with the
typed key to use when there is one. A server's synonym for a reserved field (`max_tokens` beside
`max_completion_tokens`) is the server's fact and stays free. The reserved fields are listed per
backend beside `reads_profile_key`, and a test holds the list equal to what each backend's largest
request writes: a builder that starts writing a new field makes a config that sets it refuse to
load, which is a `**Breaking:**` change.

## Schema and migrations

`src/store/migrations.rs` is an append-only ledger applied on open inside the schema lock, in one
transaction, behind an automatic backup. Four rules:

1. **Only the migration module may know an older Meka wrote the store.** No fallback readers,
   version sniffing, or "this column used to be called X" branches anywhere else. If a reader seems
   to need one, a migration is missing.
2. **A migration is frozen once any store has run it**, including a development store.
   `user_version` is a positional index, so removing or reordering an entry makes some store skip a
   step and then stamp itself current. Append; never edit a released entry. The one edit a released
   step takes is text that changes no outcome (a refusal's wording, a log line, a comment), never
   its SQL, its logic or anything it writes, and only that step's line in `SHIPPED` is replaced.
3. **A migration must be safe to run twice.** A `.dump`/restore round trip drops `user_version`, so
   steps replay over data that already has them. Guard `ALTER TABLE` on the current column set,
   prefer `IF NOT EXISTS`, and have conversions test for the shape they convert *from*. A frozen
   step names objects by the names they had, so a step that renames or drops one a frozen step reads
   must also move the floor in `classify_by_shape`, which classifies an unversioned store past every
   step before the rename by a marker the rename leaves, so no replay reaches the name that is gone.
4. **A migration may receive data it cannot work out, but may not call Meka's own code** to get it.
   Only `rusqlite`, `serde_json` and the like. A function can change meaning years later; a `String`
   cannot. `Step::Contextual` takes plain data from the caller, and its `Context` is append-only for
   the same reason the ledger is.

A new shape a store can hold moves the version even when nothing converts: a row kind, an event
variant or a column value an older build would drop or misread takes a ledger step, an empty
`Step::Sql` when there is nothing to do, so an older build refuses the store instead of reading it
wrong, as a content change moves an archive's `format_version`.

Rules 2 and 4 are enforced by `the_ledger_is_append_only` and `no_migration_calls_meka_s_own_code`;
rules 1 and 3 are not, and decay silently. The first digests *every* entry, so a legitimate append
fires it too: add the new entry's line to `SHIPPED`, never paste current values over the existing
ones.

Rule 1 has two sanctioned exceptions, both of which converge on the current shape rather than
interpreting an old one: `classify_by_shape`, which runs once per store and stamps its answer, and
`store::memory::reconcile_index`, which makes this database's FTS triggers the ones this build
requires. Rule 1 is also about *the store*, which has a ledger. `config.toml` has none and gets no
tolerance either: a reader accepts the current shape alone, never a serde alias or a parse-door
fallback. A retired or renamed key stops being modeled, `deny_unknown_fields` names it and its line,
and the upgrade guide carries the remedy. Tolerance for what a model might emit is out of scope
entirely.

A standalone migration script exists only for a conversion the migration module cannot make on its
own, such as one that must guess (which provider wrote a thinking block) and wants a human reading
the counts before it writes. It is attached to the release as an asset, never checked in. Converting
`config.toml` in a script is optional, and done only when that release ships a script for the store
anyway; a config-only change never gets one.

**Integrity guards are not compatibility.** A check that is equally true of a store created five
minutes ago defends against corruption and hand-editing; deleting it turns a fail-closed path into a
fail-open one. Keep those where the data is read.

Practical notes. The version is `PRAGMA user_version` (transactional, survives `VACUUM INTO`); never
write SQLite's unrelated `PRAGMA schema_version`. Numbers are list indices, not releases. Migration
is forward-only: downgrading means restoring the backup, and each new copy supersedes the last, so
only the most recent schema-changing upgrade is undoable. The head shape is pinned by
`HEAD_SCHEMA_FINGERPRINT` over `HEAD_TABLES`; after any schema change run
`the_head_schema_fingerprint_is_pinned`, whose failure prints the new value, and give a renamed
baseline object its new name in `BASELINE_OBJECTS` beside the old one.

## What a model reads

Every word a model reads (a tool or parameter description, a system-prompt section, a context-block
line) is paid for on every turn of every session. A description states what the thing does, its
default and what it cannot do. A refusal carries its own remedy when it happens, so a schema does
not enumerate refusals; advice belongs to standing instructions, never a schema; nothing gives
examples of what the model sees elsewhere, such as its tool list; a context-block line the model
acts on nowhere is omitted, not stated as empty; nothing is said twice. Before adding a sentence,
check whether a refusal, the tool list or another section already carries it.

The prompt cache is judged within a session: a request prefix that changes between two requests of
one session is a defect. A one-time rewrite when a new release starts is accepted, so tools and
sections are ordered by what reads logically to the model (a search ahead of the load it feeds), and
no test pins an order for the cache's sake.

## Built-in tool naming

Names are read by the model every turn and `tool_catalog` is sorted, so a name is both label and
sort key.

- **Every built-in is `<noun>_<verb>`.** The noun is the class: the config section or docs page that
  governs the tool when one exists (`shell`, `web`, `file`), else the object the tool acts on
  (`task`, `image`, `agent`), singular, qualified when a subsystem manages more than one kind of
  object (`mcp_resource`, `mcp_prompt`). It is not always the module the tool lives in. The verb is
  the operation, one per tool.
- **A trailing object only where the verb alone would read as acting on the noun itself**:
  `scratchpad_load_file` and `scratchpad_save_file`, because `load` and `save` alone would read as
  acting on the scratchpad. Accepted as names, not as a pattern.
- **Separate tight tools beat one polymorphic tool.** Arguments shape an operation and never select
  one: a tool whose operation depends on which arguments are present, or whose parameters change
  meaning with an `action` enum, is a family that wants splitting (`checklist_add`, `checklist_edit`
  and `checklist_read`, not one checklist tool). A family is fine when each member's parameters are
  distinct and self-describing.

**Renaming a tool is breaking**: names appear in config lists, user-authored skills and the history
of every existing session, so get a name right at introduction. A rename anyway takes:

- a `**Breaking:**` changelog line;
- `BUILTIN_TOOL_NAMES` (sorted), `MCP_META_TOOL_NAMES` and `builtin_primary_param` in
  `src/tools.rs`; a tool is shown by its real name on every surface, the way MCP tools are, so there
  is no display alias to update;
- the stored data it reaches: the calls in every session's history, the `denied_tools` and
  `allowed_tools` of every sub-agent spec and the tool on every background task row, each converted
  by a ledger step in the store and, in every archive an older build wrote, by an entry in the
  migration module's archive conversion (`bring_archive_forward`), the one place that may know an
  older name. The active tool set is recovered by scanning history for `tool_load` calls by name, so
  that step is what keeps a resumed session's loaded tools.

A rename anchors every substitution to a name boundary, because a blanket find-and-replace also
rewrites the MCP tool names that contain a built-in's name. The edit-distance hint
(`did_you_mean_hint`, behind `builtin_name_hint` and `near_miss_hint`) catches a typo, a dropped
`mcp__server__` prefix, a bare family noun and reordered words; a rename that changes the words
themselves leaves a resumed model with only the preamble to point it at the new name.

## Tool arguments

Every tool reads its arguments by five rules, through the readers in `src/tools/util.rs`, never a
raw `input.get(...)`. A model that fills every field it was shown sends `""` where it has nothing
to say, which a strict schema spells `null`; both mean the field was left out, and no standard
defines a blank, so Meka does, once.

- **A string.** Absent, `null` and blank (empty or whitespace alone) are "not given": omitted for an
  optional field (`optional_str`), missing for a required one (`require_str`, which refuses by
  name). A value of another type is refused naming the field, never read as absent: for a
  restriction, absent is the dangerous direction. The value comes back as sent; a parser of a word
  trims it. A field whose blank is itself a value (a replacement that deletes, a body or content
  to clear, an expiry to lift) opts out by name through the `_allowing_empty` reader, and is listed
  in the "How a built-in reads its arguments" section of `docs/book/src/tools/overview.md`.
- **A list.** `string_or_list`: a list, or one bare string as a one-entry list (a mistake models
  make constantly, and the recovery is free), or nothing. A blank entry is dropped, an entry of
  another type is refused, and `[]` stays `[]`, since an empty list is a value (the documented
  "clear" for `tags`).
- **A boolean.** `optional_bool`: `true`, `false`, or nothing; anything else, the string `"true"`
  included, is refused naming the field, since read as false it would have the call do the opposite
  of what it asked.
- **A number.** `optional_u64` (and `optional_count` over it) or `optional_i64`, through
  `text::whole_number`: an integer of the type, `2.0` included since JSON Schema's `integer` is,
  or nothing; a quoted number, a fraction or an out-of-type sign is refused naming the field,
  since read as absent it hands the call the default it did not ask for. A serde struct reads a
  number through `text::whole_number` in a deserializer of the field's own, which names the field.
  A limit past the schema's range is clamped to it; a line or message number below 1 is refused.
- **An object.** `optional_object`: an object, or nothing; anything else, the object spelled as a
  JSON string included, is refused naming the field, since read as absent it sends the call
  without what it asked for (a fetch without its `headers`).

A tool that parses through a serde struct carries the string rule with
`#[serde(default, deserialize_with = "crate::text::deserialize_blank_as_none")]` on each optional
string. `tests/layering.rs::tool_arguments_are_read_through_the_shared_readers` fails on a raw read
anywhere in `src/tools` outside the two functions its ledger names, both of which read recorded
calls among others and must tolerate what history holds; a new raw read is a new reader to write,
not a ledger entry to add.

## Background calls

A call runs in the background only when its tool says it can: `Tool::as_detachable` returns the
tool's `Detachable`, its start step. The registry offers `background` on exactly those tools and
`admit_arguments` refuses it on every other; no list of names decides it. A tool is detachable
when both hold:

- **Its call can outlast a round**: a command, a sub-agent, a fetch, a remote MCP round trip. A
  call that takes milliseconds gains nothing from detaching, and its result is wanted in the round
  that asked for it.
- **Nothing reads its effect back from its recorded result.** A detached call's recorded result
  is the started answer, and its output arrives later in a report, so a tool whose effect is
  recovered from the conversation (the checklist tools, `tool_load`, `tool_search`) would record
  an effect that has not happened. Such a tool is never detachable, however long it runs.

`start` runs inline and does everything that can refuse: it reads the arguments, resolves what the
call needs and creates what the run uses, then returns either an answer or the run. A refusal from
`start` is the call's own result and leaves no task. The run is the rest, and the guards the work
holds (a session lock, a running registration) move into it. `start` creates nothing that outlives
the process without a record: the task row is written before it, and a process the run owns is
spawned by the run. A detachable tool's `execute` is `tools::start_and_run`, so an inline call and
a background one run the same code. A run reads `ToolContext::live_call_id`, `None` once detached,
wherever it would address the call it came from, since that call has already returned.

## Layering

The tree reads top-down. `src/main.rs` dispatches to `host/` and `cli/`; the screen's slash commands
run the `cli` handlers, so `host` sits above `cli`. Both edit the installation (`config.toml`'s
accounts, profiles and MCP servers, and the credential rows) through `installation`, below them and
above `mcp`: a prompt belongs to `cli`, a request body to `host`, the door to neither. On a host
that serves the file, every write of the installation runs through `SharedDeps::write_installation`,
the door deciding on the parse it hands over, and a browser login commits through it.

`host` and `cli` both call `agent`, which calls `tools`, which calls `provider` and `mcp`, which
call `store`. `mcp` publishes its tools and never names a registry; `tools/mcp_adapter.rs` is where
they become `Tool`s. `render` is reached only from `host`, `cli` and the frontend implementations;
`streams` is the leaf that writes stderr for everyone.

`tests/layering.rs` ranks every top-level module (0 at the top) and lets a `crate::<module>` edge
point only at a strictly greater rank, further down the tree, so siblings cannot name each other.
Its tolerated list is empty; one edge is by design, `tools/subagent` reaching up to `agent`, because
a sub-agent is an agent. A new module fails the test until it is placed. Put new code where its
callers already are: a type read by `store` and `host` belongs in `store` or below, never in `host`.

Meka ships as one crate. The provider layer serves Meka's own wires and is not a library: the
Messages driver and the Claude subscription driver share one wire module, and the harness reads
`config.toml` and the store. A library crate would need a contract a party outside this repository
holds and pins, and none does; a program that wants Meka drives it over the HTTP API.
