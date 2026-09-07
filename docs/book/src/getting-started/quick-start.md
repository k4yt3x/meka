# Quick start

## 1. Add an account and a profile

Before the first run, configure an account and a profile on it. `meka account add` runs the right
credential flow (OAuth login or API-key prompt) and writes the account to
`~/.config/meka/config.toml`; `meka profile add` names the model to ask it for:

```bash
# Claude Code subscription (OAuth)
meka account add anthropic --backend claude-subscription
meka profile add work --account anthropic --model claude-opus-5

# or a Claude API key
meka account add anthropic --backend anthropic-messages
meka profile add work --account anthropic --model claude-opus-5

# or OpenAI
meka account add openai --backend openai-chat-completions
meka profile add work --account openai --model gpt-5.6-sol
```

`account add` prompts for the backend you omit and acquires the secret (browser OAuth for
`claude-subscription` / `chatgpt-subscription`, an API-key prompt otherwise), keeping it in the
store. `profile add` prompts for the account and model you omit. A sole profile is the default;
add more later and switch with `meka profile use <name>` or the per-run `--profile <name>` flag.

> If you launch `meka` with no profile configured, it errors and tells you to run `meka account add`
> and `meka profile add`. See [Configuration](../configuration/overview.md) for all options and the
> full `meka account` / `meka profile` reference.

## 2. Start using meka

After setup, you will see a prompt:

```text
meka ~/project [r] >
```

The `[r]` indicates the **read** permission level (the default). The agent can read files, search, and run shell commands in a sandbox that blocks writes. It cannot modify your files.

## 3. Ask it something

```text
meka ~/project [r] > what files are in the current directory?
```

The agent will use the `find_files` tool to list files and describe them.

## 4. Enable the workspace level

Press **Shift+Tab** to cycle the permission to `workspace`, where the agent may write inside your working directory:

```text
meka ~/project [w] >
```

Now it can modify files too, and its shell may write inside the same boundary:

```text
meka ~/project [w] > create a file called hello.txt with the text "hello world"
```

## 5. One-shot mode

For quick tasks without entering the interactive shell:

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
