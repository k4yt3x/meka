# One-Shot Mode

One-shot mode runs a single prompt and exits, similar to `bash -c`:

```bash
agsh "your prompt here"
```

The agent processes the prompt (including any tool calls), prints its response, and the process terminates. The session UUID is printed to stderr on exit.

## Examples

```bash
# Simple question
agsh "what is my current working directory?"

# File operations (requires write permission)
agsh --permission write "create a file called notes.txt with today's date"

# Search
agsh "find all TODO comments in this project"

# Web search
agsh "search the web for the latest Rust release"
```

## Combining with Other Flags

All configuration flags work in one-shot mode:

```bash
# Use a specific provider and model
agsh --provider claude -m claude-sonnet-4-20250514 "explain this codebase"

# With write permission
agsh --permission write "run 'cargo test' and summarize the results"

# Disable streaming
agsh --no-stream "read README.md and summarize it"
```

## Session Behavior

One-shot mode creates a new session for each invocation. The session UUID is printed to stderr when the run completes:

```text
Session: 550e8400-e29b-41d4-a716-446655440000
```

You can resume this session later in interactive mode:

```bash
agsh -s 550e8400-e29b-41d4-a716-446655440000
```
