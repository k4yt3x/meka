# CLI Options

```text
agsh [OPTIONS] [PROMPT]
agsh <COMMAND>
```

## Commands

### `setup`

Run the interactive configuration wizard. Prompts for provider, authentication, model, and base URL, then writes the configuration to `~/.config/agsh/config.toml`.

```bash
agsh setup
```

This wizard also runs automatically on first launch when no config file exists.

### `export`

Export a session as Markdown.

```bash
agsh export <SESSION_ID> [-o <PATH>]
```

Use `-o -` to print to stdout. See [Sessions](../usage/sessions.md#exporting-a-session) for details.

### `delete`

Delete one or more sessions by UUID, or all sessions with `--all`.

```bash
agsh delete <SESSION_ID>...
agsh delete --all
```

### `list`

List past sessions with ID, last update time, and a preview.

```bash
agsh list [-n <LIMIT>]
```

Default limit: 20.

## Arguments

### `[PROMPT]`

Run the agent's first turn immediately with this text as the user message, then drop into the interactive REPL for follow-up. Pair with [`--oneshot`](#oneshot) to exit after the first turn instead of opening the REPL.

```bash
agsh "list all files larger than 1MB in the current directory"   # first turn, then REPL
agsh --oneshot "list all files larger than 1MB"                  # first turn, then exit
```

When omitted, agsh starts the REPL with no initial input.

## Options

### `-c`, `--continue [SESSION_ID]`

Resume a session. Without a session ID, resumes the most recently updated session. With a session ID, resumes that specific session.

```bash
agsh -c                                          # resume last session
agsh -c 550e8400-e29b-41d4-a716-446655440000     # resume specific session
```

Errors if the session does not exist or is locked by another agsh instance.

### `--permission <MODE>`

Set the initial permission mode. Accepts `none` (or `n`), `read` (or `r`), `ask` (or `a`), `write` (or `w`).

```bash
agsh --permission write
agsh --permission ask
```

Default: `read`.

### `--provider <NAME>`

Set the LLM provider. Overrides `AGSH_PROVIDER` and the config file.

```bash
agsh --provider claude-oauth
```

Supported values: `openai-api`, `claude-api`, `claude-oauth`.

### `-m`, `--model <MODEL>`

Set the model name. Overrides `AGSH_MODEL` and the config file.

```bash
agsh -m gpt-4o-mini
```

### `--base-url <URL>`

Set a custom API base URL. Overrides `OPENAI_BASE_URL` and the config file.

```bash
agsh --base-url http://localhost:11434/v1
```

### `--no-stream`

Disable streaming mode. The agent waits for the complete response before displaying it. By default, responses are streamed token-by-token.

```bash
agsh --no-stream
```

### `--render-mode <MODE>`

Set the output render mode. Accepts `bat` (default), `termimad` (or `rich`), or `raw`.

- `bat`: Syntax-highlighted markdown output via bat.
- `termimad`: Full terminal formatting (box-drawn code blocks, reflowed paragraphs, formatted tables).
- `raw`: Raw markdown printed verbatim with aligned tables.

```bash
agsh --render-mode raw
```

Can also be set permanently via `display.render_mode` in the config file.

### `--thinking`

Enable extended thinking (`claude-api` and `claude-oauth` providers).

```bash
agsh --thinking
```

### `--thinking-budget <TOKENS>`

Set the extended thinking token budget. Implies `--thinking`.

```bash
agsh --thinking-budget 20000
```

### `--instructions <STRING>`

Override [`prompt.instructions`](./config-file.md#promptinstructions) for this run, replacing whatever is in the config file. Also reads from `AGSH_INSTRUCTIONS` if the flag is not given.

```bash
agsh --instructions "Be terse. No code fences in answers."
```

### `--skill <NAME>`

Invoke a [user-invocable skill](../usage/skills.md) as the first turn. Mirrors the REPL slash command [`/skill <name> [extra...]`](../usage/skills.md#invoking-a-skill-from-the-cli) — the positional `[PROMPT]` arg, if given, is prepended to the rendered skill body as additional context. Pair with [`--oneshot`](#oneshot) to exit after the turn instead of opening the REPL.

```bash
agsh --skill download-videos "https://example.com/video"             # first turn, then REPL
agsh --skill download-videos --oneshot "https://example.com/video"   # first turn, then exit
```

Errors out with a clean message if the skill name is unknown or the skill's frontmatter sets `user_invocable: false`.

### `--oneshot`

Exit after the first turn finishes. Requires either the positional `[PROMPT]` or `--skill <NAME>` — without one of those, agsh has nothing to do. Useful for scripts and CI invocations.

```bash
agsh --oneshot "summarize the last commit"
agsh --oneshot --skill deploy "to staging"
```

### `-v`, `--verbose`

Increase log verbosity. Can be repeated up to three times.

```bash
agsh -v      # info
agsh -vv     # debug
agsh -vvv    # trace
```

### `--help`

Print help information.

### `--version`

Print version information.
