# One-shot mode

One-shot mode runs a single prompt and exits, similar to `bash -c`. It takes `--oneshot` and the
prompt through `-p`:

```bash
meka --oneshot -p "your prompt here"
git diff | meka --oneshot -p -      # `-p -` reads the prompt from stdin
```

The agent processes the prompt (including any tool calls), prints its response, and the process terminates. The session id is printed to stderr on exit. A run interrupted with Ctrl+C exits 130, whatever it had printed by then.

A prompt **without** `--oneshot` is not a one-shot run: it seeds the first turn and then leaves you at the REPL prompt, which is the right default when you are working interactively and the first thing you want is already in your shell history.

`--oneshot` requires something to do, so it needs `-p` or `--skill`.

An empty or whitespace-only prompt is refused rather than sent.

[Approvals](./permissions.md#approvals) have nothing to ask from here: there is no prompt to answer, so with the switch on every tool that needs approval is refused. meka says so once at startup and names each tool as it is refused, but the run is still less useful than it looks. Give a non-interactive run the level it needs with `--permission`, or use [`meka serve`](./http-api.md) if you need a human in the loop over an API.

## Examples

```bash
# Simple question
meka --oneshot -p "what is my current working directory?"

# File operations (requires workspace permission)
meka --oneshot --permission workspace -p "create a file called notes.txt with today's date"

# Search
meka --oneshot -p "find all TODO comments in this project"

# Web page
meka --oneshot -p "summarize https://blog.rust-lang.org"
```

## Combining with other flags

All configuration flags work in one-shot mode:

```bash
# Use a specific profile
meka --oneshot --profile work -p "explain this codebase"

# With workspace permission
meka --oneshot --permission workspace -p "run 'cargo test' and summarize the results"

# Disable streaming
meka --oneshot --no-stream -p "read README.md and summarize it"

# Run one turn against an existing session
meka --oneshot -r 550e8400 -p "summarize what we decided"
```

## JSON output

`--format json` keeps stdout empty during the turn and prints one object when it ends, so a script
reads the whole turn at once rather than parsing a stream. Errors still go to stderr and the exit
code; no object is printed for a turn that failed. A turn interrupted with Ctrl+C is reported, not
failed: its object is printed with `stop_reason` set to `interrupted`, and the run exits 130 as it
does without `--format json`.

```bash
meka --oneshot -p "how many files are here?" --format json
```

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "profile": "work",
  "stop_reason": "end_turn",
  "text": "There are 14 files in this directory.",
  "tool_calls": [
    { "name": "find_files", "input": { "glob": "*" }, "is_error": false }
  ],
  "usage": {
    "input_tokens": 1180, "output_tokens": 42,
    "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1024
  },
  "notices": [
    { "level": "warn", "text": "approvals are on but nobody can answer here, so 'execute_command' was refused without asking" }
  ]
}
```

`stop_reason` is `end_turn`, `max_tokens`, `refusal` (with `refusal_text` beside it when the model
gave one) or `interrupted`. `session_id` is omitted, not `null`, for a turn interrupted before the
session existed, which is the one way a run prints a report without one; every other field is always
present. `text` is the assistant's text with the rounds joined by a blank line,
`tool_calls` lists every call in the order it was dispatched, and `usage` sums the rounds. `notices`
is what meka itself said during the turn (a refused approval, a lost write, a declined MCP
elicitation), each with
a `level` of `info` or `warn` and its `text`; it is empty when meka raised none. The flag applies to
`--oneshot` alone; a run without it is refused. `meka account usage`, `whoami` and `stats` take the
same flag and values, and `meka session export --format` takes `markdown` or `json`.

## Session behavior

One-shot mode creates a new session for each invocation, unless you point it at an existing one with `-c` (most recent) or `-r <SESSION>` (specific). Those run a single turn against that conversation and exit, which is the usual shape for scripting against a session built up earlier.

The session id is printed to stderr when the run completes:

```text
Leaving session: 550e8400-e29b-41d4-a716-446655440000
```

You can resume this session later in interactive mode:

```bash
meka -r 550e8400-e29b-41d4-a716-446655440000
```

## Piping

The answer goes to stdout and everything else to stderr, so `meka -p … 2>/dev/null | next-tool`
hands the next tool only what you asked for. That holds for every command, not just this one.

A reader that stops reading is its own decision, and meka exits `0` for it:

```bash
meka -p "summarize this log" | head -20     # exits 0; head got its lines
```

A stdout that *cannot* take the answer is a different thing, and fails the run:

```bash
meka -p "summarize this log" > /full/disk   # exits non-zero, and says why on stderr
```

The distinction matters in a script: the first is how pipelines end, the second is data you asked
for and did not get.
