# Skills

Skills are user-defined knowledge packages that give the agent non-standard knowledge -- manuals, procedures, tool-specific instructions, and experience the LLM doesn't have natively. Each skill is a directory containing a `SKILL.md` file with structured metadata.

## How Skills Work

- Skills live in `~/.config/agsh/skills/` (platform-specific config dir).
- Each skill is a directory: `skills/<name>/SKILL.md`.
- Any entry whose name begins with `.` is skipped at discovery. This covers VCS metadata (`.git`), editor/IDE state (`.vscode`, `.idea`), filesystem artifacts (`.DS_Store`, `.Trash`), and any other dotfile or dotdir that may sit alongside your skills.
- `SKILL.md` starts with a YAML frontmatter block declaring the skill's metadata, followed by Markdown body content.
- On every prompt, agsh discovers all valid skills and lists them in the system prompt with their `description` and `when_to_use`.
- The agent invokes a skill by calling the `skill` tool with the skill name. The tool returns the full body, which the agent follows.
- Skills are available in **read**, **ask**, and **write** permission modes (not in **none**).

## File Format

A skill is a directory under `~/.config/agsh/skills/` containing a `SKILL.md` file:

```
~/.config/agsh/skills/
└── download-videos/
    └── SKILL.md
```

`SKILL.md` must begin with a YAML frontmatter block, followed by the skill body:

```markdown
---
description: Download videos from various websites using yt-dlp
when_to_use: When the user wants to download a video from a website
allowed_tools: [execute_command]
version: "1.0"
user_invocable: true
---

# Download Videos with yt-dlp

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
```

### Required Frontmatter Fields

| Field | Description |
|-------|-------------|
| `description` | One-line summary of what the skill does. Shown in the system prompt. |
| `when_to_use` | A hint telling the agent when to invoke the skill. Shown in the system prompt. |

Skills missing either field are skipped at discovery with a warning log.

### Optional Frontmatter Fields

| Field | Default | Description |
|-------|---------|-------------|
| `allowed_tools` | `[]` | Array or CSV string of tool names the skill expects. Currently advisory (not enforced). |
| `version` | none | Free-form version label (e.g. `"1.0"`, `"2024-03-14"`). |
| `user_invocable` | `true` | Reserved for future `/skill <name>` slash command. |

### Variable Substitution

The skill body may reference these variables, which are expanded when the skill is loaded:

- `${AGSH_SKILL_DIR}` -- the absolute path to the skill's directory. Use this to reference bundled helper files (e.g. `${AGSH_SKILL_DIR}/helper.sh`).
- `${AGSH_SESSION_ID}` -- the current session UUID.

## Storage Location

| Platform | Path |
|----------|------|
| Linux | `~/.config/agsh/skills/<name>/SKILL.md` (`$XDG_CONFIG_HOME/agsh/skills/`) |
| macOS | `~/Library/Application Support/agsh/skills/<name>/SKILL.md` |
| Windows | `%APPDATA%\agsh\skills\<name>\SKILL.md` |

## How the Agent Uses Skills

When skills are available, the system prompt includes a `## Skills` section like:

```
## Skills

- **download-videos**: Download videos from various websites using yt-dlp — When the user wants to download a video from a website
- **deploy-kubernetes**: Deploy services to a K8s cluster — When the user asks to deploy to Kubernetes
```

The agent loads a skill by calling the `skill` tool:

```
skill(name: "download-videos")
```

The tool returns the full body of `SKILL.md` (with variables expanded) as its output. The agent then follows the instructions.

## Invoking a Skill from the CLI

Skills marked `user_invocable: true` (the default) can also be triggered directly from the command line with `--skill <name>`. The rendered body becomes the first user turn, and agsh drops into the interactive REPL after the turn finishes:

```bash
agsh --skill download-videos "https://example.com/video"
```

The positional `[PROMPT]` argument, if given, is prepended to the skill body as extra context — equivalent to typing `/skill download-videos https://example.com/video` in the REPL.

To run the skill and exit immediately (useful for scripts), pair with `--oneshot`:

```bash
agsh --oneshot --skill download-videos "https://example.com/video"
```

To invoke a skill mid-session inside the REPL, use the slash command instead:

```
/skill download-videos
/skill download-videos this URL specifically
```

## Tips

- Use short, unambiguous skill names (e.g. `setup-postgres`, not `pg`). The name is what the agent sees and calls.
- Write `description` and `when_to_use` concisely -- they go into every system prompt and consume tokens.
- Keep each skill focused on a single topic or procedure. Spawn multiple skills rather than one giant one.
- Bundle supporting files in the skill directory and reference them with `${AGSH_SKILL_DIR}/file.ext`.
- Skills are re-discovered on every prompt, so you can add, edit, or remove skills mid-session without restarting agsh.
