# Skills

Skills are user-defined knowledge packages that give the agent non-standard knowledge -- manuals, procedures, tool-specific instructions, and experience the LLM doesn't have natively. Each skill is a directory containing a `SKILL.md` file with structured metadata.

## How Skills Work

- Skills live in `~/.config/agsh/skills/` (platform-specific config dir).
- Each skill is a directory: `skills/<name>/SKILL.md`.
- Any entry whose name begins with `.` is skipped at discovery. This covers VCS metadata (`.git`), editor/IDE state (`.vscode`, `.idea`), filesystem artifacts (`.DS_Store`, `.Trash`), and any other dotfile or dotdir that may sit alongside your skills.
- `SKILL.md` starts with a YAML frontmatter block declaring the skill's metadata, followed by Markdown body content.
- On every prompt, agsh discovers all valid skills and lists them in the system prompt with their `description`.
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
description: Download videos from various websites using yt-dlp. Use when the user wants a video off a URL.
version: "1.0"
author: John Doe <john.doe@example.com>
source_url: https://gist.githubusercontent.com/k4yt3x/.../raw/SKILL.md
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
| `description` | Summary of what the skill does *and when to invoke it*. Shown in the system prompt — fold the trigger condition into this one line. |

Skills missing `description` are skipped at discovery with a warning log. Unknown frontmatter keys are ignored, so a skill authored for Claude Code (which carries extra keys like `when_to_use` or `allowed-tools`) still loads.

### Optional Frontmatter Fields

| Field | Default | Description |
|-------|---------|-------------|
| `version` | none | Free-form version label (e.g. `"1.0"`, `"2024-03-14"`). |
| `author` | none | Attribution, conventionally `Name <email>` (e.g. `John Doe <john.doe@example.com>`). Informational only. |
| `source_url` | none | An `https://` URL the skill's `SKILL.md` can be re-fetched from. Enables [`agsh skill update`](#updating-skills). |

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

- **download-videos**: Download videos from various websites using yt-dlp. Use when the user wants a video off a URL.
- **deploy-kubernetes**: Deploy services to a K8s cluster. Use when the user asks to deploy to Kubernetes.
```

The agent loads a skill by calling the `skill` tool:

```
skill(name: "download-videos")
```

The tool returns the full body of `SKILL.md` (with variables expanded) as its output. The agent then follows the instructions.

## Invoking a Skill from the CLI

Any skill can be triggered directly from the command line with `--skill <name>`. The rendered body becomes the first user turn, and agsh drops into the interactive REPL after the turn finishes:

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

## Updating Skills

A skill that declares a `source_url` can be re-fetched and replaced on disk with `agsh skill update`:

```bash
agsh skill update download-videos   # update one skill
agsh skill update --all             # dry run: lists what would update
agsh skill update --all --yes       # apply the updates
```

`source_url` should be an `https://` link to a raw `SKILL.md` (e.g. a GitHub raw URL or a gist raw URL). The fetch is validated — the response must parse as a valid skill — before the on-disk file is atomically replaced, so a 404 page or a malformed file leaves the existing skill untouched. If the fetched content is byte-identical to what's on disk, nothing is written.

`agsh skill update --all` without `--yes` is a dry run: it lists every skill that would be updated and applies nothing. This is the confirmation gate for a bulk remote fetch — re-run with `--yes` to apply.

Only the `SKILL.md` file is fetched. Helper scripts bundled alongside it in the skill directory are **not** updated this way — `source_url`-based update is for single-file skills.

> **Trust note.** A skill body is instructions the agent follows. `agsh skill update` replaces that content with whatever the `source_url` currently serves — review the source you point it at, and prefer `--all` (with its dry-run default) over blind updates.

## Tips

- Use short, unambiguous skill names (e.g. `setup-postgres`, not `pg`). The name is what the agent sees and calls.
- Write `description` concisely, and fold the "use when..." trigger into it -- it goes into every system prompt and consumes tokens.
- Keep each skill focused on a single topic or procedure. Spawn multiple skills rather than one giant one.
- Bundle supporting files in the skill directory and reference them with `${AGSH_SKILL_DIR}/file.ext`.
- Skills are re-discovered on every prompt, so you can add, edit, or remove skills mid-session without restarting agsh.
