# Skills

Skills are user-defined knowledge files that provide the agent with non-standard knowledge -- manuals, procedures, tool-specific instructions, and experiences that the LLM doesn't have natively.

## How Skills Work

- Skills are Markdown files stored in `~/.config/agsh/skills/` (or the platform-specific config directory).
- On every prompt, agsh scans the skills directory and lists available skills in the system prompt.
- The agent can then read any skill on demand using the `read_file` tool.
- Skills are available in **read** and **write** permission modes.

## File Format

Each skill file should follow this structure:

1. **Title** on the first line (a Markdown heading)
2. **Blank line**
3. **Summary paragraph** describing what the skill covers

The title and summary allow the agent to quickly preview what a skill is about without reading the full file.

### Example

`~/.config/agsh/skills/download-videos.md`:

```markdown
# Download Videos with yt-dlp

A guide to downloading videos from various websites using yt-dlp.

## Installation

Install yt-dlp:
\```bash
pip install yt-dlp
\```

## Basic Usage

Download a video:
\```bash
yt-dlp "https://example.com/video"
\```

## Common Options

- `-f bestvideo+bestaudio` -- select best quality
- `-o "%(title)s.%(ext)s"` -- set output filename template
- `--sub-langs en` -- download English subtitles
```

## Storage Location

| Platform | Path |
|----------|------|
| Linux | `~/.config/agsh/skills/` (`$XDG_CONFIG_HOME/agsh/skills/`) |
| macOS | `~/Library/Application Support/agsh/skills/` |
| Windows | `%APPDATA%\agsh\skills\` |

## How the Agent Uses Skills

When skills are available, the system prompt includes a `## Skills` section listing all `*.md` files in the skills directory. The agent can:

1. **Preview** a skill by reading the first few lines:
   ```
   read_file(path: "~/.config/agsh/skills/download-videos.md", limit: 3)
   ```

2. **Read the full skill** when it needs detailed instructions:
   ```
   read_file(path: "~/.config/agsh/skills/download-videos.md")
   ```

## Tips

- Use descriptive file names (e.g., `deploy-kubernetes.md`, `setup-postgres.md`) so the agent can identify relevant skills from the listing alone.
- Keep skills focused on a single topic or procedure.
- Skills are re-scanned on every prompt, so you can add, update, or remove skills mid-session.
