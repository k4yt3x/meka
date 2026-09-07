# AGENTS.md

Guidance for AI agents working in this repository. Three sections: general principles, Rust
practice, then meka-specific rules.

---

# General principles

## Code

- Correctness and clarity first. Speed and efficiency are secondary unless stated otherwise.
- Comments explain *why*, never *what*: a constraint from outside this file, or why an
  obvious-looking alternative is wrong. Not history, not user-facing documentation, not argument for
  the choice; `git log` and `docs/book/src/` hold those. If it restates the code, delete it.
- Add functionality to existing files unless it is genuinely a new component. Avoid many small files.
- No creative additions beyond what was asked.
- Full words in names, no abbreviations.

## Enumerate the doors

Most defects that survive review are one rule enforced at one entry point and not at its siblings.

Before writing a guard, list every path that reaches the thing being guarded (create, copy, fork,
import, re-root, resume, re-attach, patch, delete) and decide for each. Place the check where those
paths converge. A rule placed at a door is a rule the next door forgets.

**Prefer one definition over many checks.** When an invariant is asked in more than one place, give it
a single named predicate and call that everywhere. Consolidation beats a test per site, because tests
only confirm the sites you thought of.

**A guard sits ahead of every side effect it protects**, not merely ahead of the failure. Refusing
after a write leaves the write behind.

## Verification

Verification is graded by what it catches, not by how much of it there is.

**Per change**: build, test suite, and the project's exact CI lint gate. Then:

- **Fake-guard every test written to protect a fix**: neuter the fix, confirm the test fails, restore.
  A test that cannot fail is worse than no test, because it reads as coverage.
- Enumerate the doors, as above.

**Per release**: one structural review, docs and changelog. Don't run a cross-platform suite by hand
per change; CI already runs the matrix on every push.

**Rarely**: mutation testing. Run it when a subsystem is new, not as a gate; its yield falls as the
suite densifies while its cost grows with the codebase.

Reproduce a defect before fixing it, and re-run the reproduction after. A fix verified only by a
passing suite was verified against the thing that already missed it.

Do not grow the suite reflexively. A test earns its place by being able to fail.

## Read the code, not the comment

A doc comment is a claim about the code, not evidence for it. Where a comment and its code disagree,
the comment is usually what was updated last and least. Verify a stated invariant against the
implementation before relying on it, and correct the comment when it is wrong.

## Whose fact is it

If another system is the authority for a fact, ask it or let the user state it. Never encode it.
A hardcoded fact about an external system expires, and nothing in the build notices.

- **The provider/service owns it**: a request parameter it defaults sensibly. Omit it unless the user
  asked for a value; omitting *is* how you request their default.
- **The user owns it**: anything neither side can determine, or where a wrong guess is invisible. One
  config key, one documented default. Don't infer, probe, or cache. State that default in the docs
  and, where a setup flow exists, on screen.
- **We own it**: our own names, schema, and defaults for our own behavior. Encoding these is fine.

A guess is tolerable when its wrong answer is a *rejected request* and it fails toward omission. A
guess that fails toward *sending* survives only where the endpoint cannot vary, so never introduce one
on a backend reachable via a user-supplied `base_url`. A value verified against a captured wire is a
fact about the protocol rather than a prediction; pin it deliberately and cite the capture.

## Compatibility

Backwards compatibility spread across readers costs one shim per reader per superseded shape, forever.
Convert once, in one place, so every other reader may assume the current shape unconditionally. That
assumption is the entire return, and it is lost the moment a second place tolerates an old shape.

## Changelog

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
  hold the detail and the entry names the page. Trivial changes are compacted into one line or
  left out.
- Breaking changes get an inline `**Breaking:**` prefix inside their type, not a separate section.

## Prose style

- American spelling everywhere meka owns the word: prose, comments, docs, strings, identifiers and
  meka's own wire values (`canceled`, `color`, `catalog`, `behavior`). Only a name another protocol
  or crate defines keeps its spelling (ACP's `stopReason: "cancelled"`, MCP's
  `notifications/cancelled`, `serde::Serialize`), and a value a model emits may be accepted in both
  spellings while meka writes one.
- No em dashes (`—`). Prefer a colon, a comma or parentheses.
- Sentence-case headings in the book; product names keep their case.

---

# Rust practice

## Safety

- Avoid panicking calls (`unwrap()`, `expect()`, unchecked indexing). Propagate with `?`. The lints
  are `warn` in `Cargo.toml` and relaxed under `cfg(test)`, where panicking on failure is the point.
- Never discard errors with `let _ =`. Propagate with `?`, log explicitly when ignoring is correct, or
  handle with `match` / `if let Err(..)`.
- Errors from fallible async work must reach the UI layer so the user gets real feedback.

## Layout and style

- No `mod.rs`. Use `src/some_module.rs`.
- New crates set `[lib] path = "..."` in `Cargo.toml` for a descriptive root name.
- Never hand-wrap comments. One line per paragraph; `cargo +nightly fmt` wraps them
  (`.rustfmt.toml` sets `wrap_comments = true`). If a wrap lands awkwardly, reword rather than
  inserting a manual break.
- Shadow a binding to scope a clone in async contexts:

  ```rust
  executor.spawn({
      let task_ran = task_ran.clone();
      async move { *task_ran.borrow_mut() = true; }
  });
  ```

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
  Values a model emits (todo status words) are the one exception.

## Build gate

Run after editing: `cargo +nightly fmt` and `cargo sort -w`.

CI denies warnings on clippy and rustdoc, so the bare commands can pass locally and fail CI.
Reproduce the exact gate before declaring done:

```
cargo +nightly fmt --check
cargo sort -w --check
cargo clippy --locked --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --document-private-items
cargo test --locked                  # CI adds --features mock-provider; debug builds carry it anyway
cargo check --locked --all-targets   # on the MSRV in Cargo.toml's rust-version
mdbook build docs/book
```

`--all-targets` matters: plain clippy skips tests and benches. In rustdoc, watch
`rustdoc::invalid_html_tags`: a bare `<word>` parses as an unclosed tag. Backtick it, or rephrase if
the comment is also a clap help string, where backticks render literally.

`fmt --check` does not enforce `comment_width`: `wrap_comments` silently declines some comments (in
a macro body, in a method chain) and still exits 0, so a paragraph left for rustfmt to wrap can ship
at 400 columns. After `fmt`, `awk 'length>100 && /^[[:space:]]*\/\//'` the changed files; reword
what it prints, or break it by hand.

## Clap help text

`///` doc comments must render within 120 columns under `-h`, and stay as short as they can: no
examples, no tautology, one line where one line says it. Verify by running the binary with
`COLUMNS=120` for every changed subcommand: source length ignores clap's indent, value-name width,
and auto-appended hints. Adding flags widens the whole column, so a new flag can push existing lines
over.

Long-form prose goes after a blank `///` line so it appears only under `--help`. When that prose is
multi-line or indented, add `#[command(verbatim_doc_comment)]`.

Command summaries take no trailing period; multi-sentence prose is punctuated normally.

---

# meka

## Vocabulary

One word per concept, everywhere it is written or read:

- **permission level**, never "mode": `level` in prose, `permission` in identifiers. ACP's own
  `set_mode` and `mode_id` keep their protocol names.
- **profile**, **account**, **backend**. "Provider" is only the upstream service and the `Provider`
  trait: "provider temporarily unavailable" is right, "provider profile" is not.
- **store** for the `Store`: `store` in identifiers, "the store" in prose, `meka.db` for the file.
  `manager` is only `McpClientManager`.
- **sub-agent** as the noun; "worker" only when the orchestration role is the point; never
  "delegate" as a noun. Depth 0 is the **root**, the spawner is the **parent**.
- **scratchpad entry** in every string and identifier; the SQL table `tool_outputs` keeps its name.
- **approvals** is the switch; the prompt is an **approval prompt** with the header `[approval]`;
  its answers are allow and deny.
- **standing instructions** for the concept; "instructions file" only for the file on disk.
- **id** lowercase; "UUID" only where the format is the point.
- Refusal verbs: *refuse* is meka's own decision; *reject* is a remote party (a provider, an MCP
  server); *deny* is the OS, a permission, or the user's answer; *decline* is a gate or the model.
- Quoting in a message: backticks for keys, commands and code (`` `[permissions].default` ``,
  `` `meka profile use` ``); single quotes for values and names (`'work'`). An unknown name is
  reported through `text::unknown_name`, never a per-door sentence.
- An empty list says "No <nouns>." on stderr. A status line that is a full sentence ends in a
  period ("Connected to 'exa'."); one that ends in a value, id or path does not ("Profile set
  to: work").
- A message the user reads (error, warning, notice, hint, help) says what happened and, when there
  is exactly one, the remedy: no examples, no alternatives, no explanation of internals, no
  second sentence that restates the first. One sentence where one suffices. A printed line stays
  within 120 columns where its content allows, so it fits a terminal without wrapping.

## Output: prints vs. tracing

**If the user doesn't have to see it to use the command, it is a log.** Default level is `warn`, so
`info!` / `debug!` are silent unless the user passes `-v`, `-vv`, or `RUST_LOG`. Aim for quiet on
success.

`println!` / `eprintln!` only for: requested data; content the user must copy, type, or visit; REPL
command output; and hard errors. Everything else is `tracing`.

The stream is a contract:

- **stdout**: only the data the command was invoked to obtain. In the REPL that is the model's
  answers (and the line editor, which draws there); every slash command's output, tables included,
  is stderr.
- **stderr**: everything else, including prompts, live UI, indicators, hints, status and errors,
  and every spacing blank line emitted around them.

Litmus test: `meka ... 2>/dev/null | next-tool` must leave only the requested data on stdout.

Levels: `error!` for an unrecoverable failure about to propagate or a turn whose outcome is lost;
`warn!` for a recoverable fallback, a rollback, or a lost write the user can act on (never `debug!`
for one); `info!` for lifecycle signposts; `debug!` for module-level diagnostics. Every log string
uses inline captures (`{name}`), and a failure reads "failed to <verb>", never "could not".

Don't invert it either: a command's primary output must not be a `tracing::info!`, or the user needs
`-v` to see what they asked for. `ok:` confirmations are logs, not prints; the exit code carries
success. Drop preambles before the actionable line. Honor a config flag that asks for visible
output; don't demote it to `info!`.

## Configuration surfaces

- **`config.toml` is the complete source of truth** for non-secret settings. Every persistent
  setting lives there.
- **Accounts and profiles are config-only, never env.** An ambient variable must never silently
  rebind which account a named profile bills. An account (`[accounts.<name>]`) is a backend, an
  endpoint and the credential a login produced; a profile (`[profiles.<name>]`) is an account plus a
  model and every model-tied setting. A session runs on the profile its own row names; the row moves
  only by an explicit act (`--profile` on a resume, `/profile`, `PATCH /v1/sessions/{id}`, ACP
  `session/set_config_option`). What a *new* session records follows `--profile` > `default_profile`
  > the sole profile. Accounts are managed by `meka account` (`add`/`login`/`list`/`remove`) and
  profiles by `meka profile` (`add`/`set`/`use`/`list`/`remove`), mirroring `meka mcp`: `use` is the
  only command that sets `default_profile` (`profile remove` unsets it when it named the removed
  profile), `account login` rotates a credential without touching the account or its profiles, and
  `account remove` refuses while a profile names the account.
- **Secrets live in the database**: `account_credentials` keyed by account name and
  `mcp_credentials` keyed by `(server_name, kind)`, so two accounts, or a client secret and its
  refreshable bundle, can coexist. Every secret is read from stdin, never taken
  as an argument, because arguments are visible in `ps` and shell history.
  - A field that may *contain* a secret is not itself one; it stays in `config.toml` with `${VAR}`
    expansion.
  - Retiring a config key that held a secret gets no compatibility shim. It stops being modeled and
    `deny_unknown_fields` names the key and line; the upgrade guide carries the remedy.
- **Environment variables are operational only**: `MEKA_CONFIG_DIR`, `MEKA_DATA_DIR`, permission,
  instructions, sandbox backend, render mode, MCP tool timeout, `RUST_LOG`. Precedence is CLI > env
  > file, written as `cli.x.or_else(env).or(file)` in `ResolvedConfig::resolve`.
- **Session and display tuning is config-only.** No env vars or flags for set-once preferences.
  Render mode is the one exception, because the program that launches meka, not the user, knows
  whether its output is a terminal.

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
  profile onto another credential; `device_id` is not settable at all, because meka resolves it.
- **`backend` names the driver**, not a protocol: API-key backends are named for the protocol
  because `base_url` decides the endpoint, subscription backends for the product because the
  endpoint is fixed.

## Schema and migrations

`src/store/migrations.rs` is an append-only ledger applied on open inside the schema lock, in one
transaction, behind an automatic backup. Four rules:

1. **Only the migration module may know an older meka wrote the store.** No fallback readers, version
   sniffing, or "this column used to be called X" branches anywhere else. If a reader seems to need
   one, a migration is missing.
2. **A migration is frozen once any store has run it**, including a development store. `user_version`
   is a positional index, so removing or reordering an entry makes some store skip a step and then
   stamp itself current. Append; never edit a released entry.
3. **A migration must be safe to run twice.** A `.dump`/restore round trip drops `user_version`, so
   steps replay over data that already has them. Guard `ALTER TABLE` on the current column set, prefer
   `IF NOT EXISTS`, and have conversions test for the shape they convert *from*.
4. **A migration may receive data it cannot work out, but may not call meka's own code** to get it.
   Only `rusqlite`, `serde_json` and the like. A function can change meaning years later; a `String`
   cannot. `Step::Contextual` takes plain data from the caller, and its `Context` is append-only for
   the same reason the ledger is.

Rules 2 and 4 are enforced by `the_ledger_is_append_only` and `no_migration_calls_meka_s_own_code`;
rules 1 and 3 are not, and decay silently. The first digests *every* entry, so a legitimate append
fires it too: add the new entry's line to the expected vector, never paste current values over the
existing ones.

Rule 1 has two sanctioned exceptions, both of which converge on the current shape rather than
interpreting an old one: `classify_by_shape`, which runs once per store and stamps its answer, and
`store::memory::reconcile_index`, which makes this database's FTS triggers the ones this build
requires. Rule 1 is also about *the store*, which has a ledger. `config.toml` has none and gets no
tolerance either: a shape change ships with a one-shot conversion script attached to the release, never
a serde alias or a parse-door fallback. Tolerance for what a model might emit is out of scope entirely.

**Integrity guards are not compatibility.** A check that is equally true of a store created five
minutes ago defends against corruption and hand-editing; deleting it turns a fail-closed path into a
fail-open one. Keep those where the data is read.

Practical notes. The version is `PRAGMA user_version` (transactional, survives `VACUUM INTO`); never
write SQLite's unrelated `PRAGMA schema_version`. Numbers are list indices, not releases. Migration is
forward-only: downgrading means restoring the backup, and each new copy supersedes the last, so only
the most recent schema-changing upgrade is undoable.

## Built-in tool naming

Names are read by the model every turn and `tool_catalog` is sorted, so a name is both label and
sort key.

- **A family shares a noun prefix**: `<subsystem>_<verb>`, which is what makes the family arrive as
  one sorted block. It names what the tools act on, which is not always the module they live in.
  Where a subsystem manages more than one kind of object, qualify before the verb and keep the object
  first. A verb that merely mentions a noun does not make it a managed object.
- **A standalone tool reads as a verb phrase**: `<verb>_<object>`. A subsystem with one operation may
  use the bare noun.

Two exceptions. **An industry-standard name beats internal consistency**: models reach for
`read_file`, `write_file`, `edit_file` and `execute_command` zero-shot, and renaming them trades
accuracy for tidiness. And **`load_tool` stays verb-first** despite acting on meka's own registry,
because the name appears verbatim in the `[Tool discovery]` preamble the model reads every turn.
`scratchpad_load_file` and `scratchpad_save_file` carry a trailing object because `load` and `save`
alone would read as acting on the scratchpad itself; accepted as names, not as a pattern.

Renaming a tool is breaking: names appear in config lists, user-authored skills, and the history of
every existing session. Prefer getting it right at introduction. When renaming anyway, add a
`**Breaking:**` changelog line and update `BUILTIN_TOOL_NAMES` (sorted), `MCP_META_TOOL_NAMES`, and
`builtin_primary_param` in `src/tools.rs`. A tool is shown by its real name on every surface, the
way MCP tools are; there is no display alias to update. Two silent traps: a blanket
find-and-replace rewrites MCP tool names containing a built-in as a substring, so anchor every
substitution to a name boundary; and reversing word order defeats the edit-distance hint
(`did_you_mean_hint`, behind `builtin_name_hint` and `near_miss_hint`), so nothing points a resumed
model at the new name.

## Layering

The tree reads top-down. `src/main.rs` dispatches to `host/` and `cli/`; the REPL's slash commands
run the `cli` handlers, so `host` sits above `cli`. Both call `agent`, which calls `tools`, which
calls `provider` and `mcp`, which call `store`. `mcp` publishes its tools and never names a registry;
`tools/mcp_adapter.rs` is where they become `Tool`s. `render` and `console` are reached only from
`host`, `cli` and the frontend implementations; `streams` is the leaf that writes stderr for
everyone. `tests/layering.rs` ranks every top-level module and lets a `crate::<module>` edge point
only at a strictly lower rank, so siblings cannot name each other. Its tolerated list is empty; one
edge is by design, `tools/subagent` reaching up to `agent`, because a sub-agent is an agent. A new
module fails the test until it is placed. Put new code where its callers already are: a type
read by `store` and `host` belongs in `store` or below, never in `host`.

## Documentation

Update the mdBook docs under `docs/book/src/` for any user-facing change, and the upgrade guide for
anything marked `**Breaking:**`.
