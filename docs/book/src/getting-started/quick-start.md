# Quick Start

## 1. Run the Setup Wizard

On first launch, agsh automatically starts an interactive setup wizard:

```bash
agsh
```

The wizard will guide you through:

1. **Provider selection** — Choose between `claude` and `openai`
2. **Authentication** — OAuth login (Claude only) or API key entry
3. **Model selection** — Enter the model name to use
4. **Base URL** — Optionally set a custom API endpoint

The wizard writes your configuration to `~/.config/agsh/config.toml`. You can re-run it at any time with `agsh setup`.

> You can also create the config file manually or use environment variables (`OPENAI_API_KEY`, `AGSH_PROVIDER`, etc.) and CLI flags (`--provider`, `-m`) as overrides. See [Configuration](../configuration/overview.md) for all options.

## 2. Start Using agsh

After setup, you will see a prompt:

```text
agsh [r] >
```

You will see a prompt:

```text
agsh [r] >
```

The `[r]` indicates **read** permission mode (the default). The agent can read files and search, but cannot write files or run commands.

## 3. Ask It Something

```text
agsh [r] > what files are in the current directory?
```

The agent will use the `find_files` tool to list files and describe them.

## 4. Enable Write Mode

Press **Shift+Tab** to cycle the permission to write mode:

```text
agsh [w] >
```

Now the agent can execute commands and modify files:

```text
agsh [w] > create a file called hello.txt with the text "hello world"
```

## 5. One-Shot Mode

For quick tasks without entering the interactive shell:

```bash
agsh "what is my current working directory?"
```

The process exits after the agent responds.

## 6. Continue a Previous Session

To pick up where you left off, continue the last session:

```bash
agsh -c
```

Or resume a specific session by its UUID:

```bash
agsh -c 550e8400-e29b-41d4-a716-446655440000
```

See [Sessions](../usage/sessions.md) for more details.
