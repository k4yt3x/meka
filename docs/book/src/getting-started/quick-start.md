# Quick start

## 1. Add an account and a profile

Before the first run, configure an account and a profile on it. `meka account add` runs the right
credential flow (OAuth login or API-key prompt) and writes the account to
`~/.config/meka/config.toml`; `meka profile add` names the model to ask it for:

```bash
# Claude Code subscription (OAuth)
meka account add claude --backend claude-subscription
meka profile add work --account claude --model claude-opus-5-5

# or a Claude API key
meka account add anthropic --backend anthropic-messages
meka profile add work --account anthropic --model claude-opus-5-5

# or OpenAI
meka account add openai --backend openai-chat-completions
meka profile add work --account openai --model gpt-6-astra
```

`account add` prompts for the backend you omit and acquires the secret (browser OAuth for
`claude-subscription` / `chatgpt-subscription`, an API-key prompt otherwise), keeping it in the
store. `profile add` prompts for the account and model you omit. A sole profile is the default;
add more later and switch with `meka profile use <name>` or the per-run `--profile <name>` flag.

> If you launch `meka` with no profile configured, it errors and tells you to run `meka account add`
> and `meka profile add`. See [Configuration](../configuration/overview.md) for all options and the
> full `meka account` / `meka profile` reference.

## 2. Start using meka

After setup, run `meka`. The screen opens on the conversation, with a composer at the bottom and a status row under it that opens with the permission level: **read** by default. The agent can read files, search, and run shell commands in a sandbox that blocks writes. It cannot modify your files.

## 3. Ask it something

```text
> what files are in the current directory?
```

The agent will use the `file_find` tool to list files and describe them.

## 4. Enable the workspace level

Press **Shift+Tab** to cycle the permission to `workspace`, where the agent may write inside your working directory. Now it can modify files too, and its shell may write inside the same boundary:

```text
> create a file called hello.txt with the text "hello world"
```

## 5. One-shot mode

For quick tasks without opening the screen:

```bash
meka --oneshot -p "what is my current working directory?"
```

The process exits after the agent responds. Without `--oneshot` the same prompt runs as the first
turn and then drops you into the interactive shell.

## 6. Continue a previous session

To pick up where you left off, continue the last session:

```bash
meka -c
```

Or resume a specific session by its id:

```bash
meka -r 550e8400-e29b-41d4-a716-446655440000
```

See [Sessions](../usage/sessions.md) for more details.
