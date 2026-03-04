# Quick Start

## 1. Create a Config File

Create `~/.config/agsh/config.toml` with your provider settings:

**OpenAI:**

```toml
[provider]
name = "openai"
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-or-v1-..."
model = "anthropic/claude-opus-4.6"
```

**Anthropic:**

```toml
[provider]
name = "anthropic"
model = "claude-opus-4.6"
api_key = "sk-ant-..."
```

See [Configuration](../configuration/overview.md) for all options including custom endpoints.

> You can also use environment variables (`OPENAI_API_KEY`, `AGSH_PROVIDER`, etc.) or CLI flags (`--provider`, `-m`) as overrides. See [Environment Variables](../configuration/environment-variables.md) and [CLI Options](../configuration/cli-options.md).

## 2. Start the Shell

```bash
agsh
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
agsh -p "what is my current working directory?"
```

The process exits after the agent responds.

## 6. Continue a Previous Session

To pick up where you left off, continue the last session:

```bash
agsh -c
```

Or resume a specific session by its UUID:

```bash
agsh -s 550e8400-e29b-41d4-a716-446655440000
```

See [Sessions](../usage/sessions.md) for more details.
