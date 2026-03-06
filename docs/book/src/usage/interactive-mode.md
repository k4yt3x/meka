# Interactive Mode

Start agsh without the `-p` flag to enter interactive mode:

```bash
agsh
```

You get a prompt:

```text
agsh [r] >
```

Type your instruction and press **Enter** to submit. The agent processes your request and prints its response (streamed in real time as Markdown). When it finishes, you get another prompt.

## Keybindings

agsh uses Emacs-style keybindings (provided by reedline).

### Input

| Key | Action |
|-----|--------|
| Enter | Submit the current prompt |
| Alt+Enter | Insert a newline (for multi-line input) |
| Shift+Tab | Cycle the permission mode (none &rarr; read &rarr; write &rarr; none) |

### Navigation

| Key | Action |
|-----|--------|
| Ctrl+A | Move cursor to start of line |
| Ctrl+E | Move cursor to end of line |
| Ctrl+F | Move cursor forward one character |
| Ctrl+B | Move cursor backward one character |
| Alt+F | Move cursor forward one word |
| Alt+B | Move cursor backward one word |

### Editing

| Key | Action |
|-----|--------|
| Ctrl+D | Delete character under cursor / exit on empty line |
| Ctrl+H, Backspace | Delete character before cursor |
| Ctrl+K | Kill text from cursor to end of line |
| Ctrl+U | Kill text from start of line to cursor |
| Ctrl+W | Kill word before cursor |
| Ctrl+Y | Yank (paste) killed text |

### Control

| Key | Action |
|-----|--------|
| Ctrl+C | Interrupt the running agent; clear the line if idle |
| Ctrl+D | Exit the shell (when the line is empty) |
| Ctrl+R | Reverse incremental search through history |
| Ctrl+L | Clear the screen |

## Prompt Format

```text
agsh [indicator] >
```

The indicator shows the current permission mode:

| Mode | Indicator | Color |
|------|-----------|-------|
| None | `[n]` | Green |
| Read | `[r]` | Yellow |
| Write | `[w]` | Red |

The color provides a visual cue about the agent's current capabilities. Red means the agent can modify your system.

## Multi-Line Input

Press **Alt+Enter** to insert a newline instead of submitting. The prompt changes to show continuation:

```text
agsh [r] > write a python script that
  ... prints hello world
  ... and saves it to hello.py
```

Press **Enter** on the last line to submit the entire multi-line input.

## Slash Commands

agsh supports `/` prefix commands for controlling the shell:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit` | Exit the shell |
| `/clear` | Clear the terminal screen |
| `/session` | Show the current session ID |
| `/permission [none\|read\|write]` | Show or set the permission level |
| `/compact` | Summarize and compact the session history |

### `/compact`

The `/compact` command asks the LLM to summarize the entire conversation, then replaces the message history with a single summary message. This is useful for long sessions that are approaching the context window limit or becoming expensive.

After compacting, the session continues with the summary as context. The previous messages are removed from both memory and the database.

## Shell Escape

Prefix any input with `!` to execute it directly as a shell command, bypassing the LLM entirely:

```text
agsh [r] > !pwd
/home/user/projects
agsh [r] > !ls -la
total 32
drwxr-xr-x  5 user user 4096 Mar  4 10:00 .
...
agsh [r] > !ping 1.1.1.1 -c 2
PING 1.1.1.1 (1.1.1.1) 56(84) bytes of data.
...
```

The command runs with inherited stdin/stdout/stderr, so it behaves exactly like a regular shell. This is useful for quick checks without waiting for the LLM.

## Exiting

You can exit agsh in any of these ways:

- Type `/exit`
- Type `exit` or `quit`
- Press **Ctrl+D** on an empty line

## Interrupting the Agent

Press **Ctrl+C** while the agent is running to interrupt it. This cancels the current LLM request and kills any running shell commands that were spawned by the agent.
