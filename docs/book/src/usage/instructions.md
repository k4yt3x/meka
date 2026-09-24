# Instructions

Standing instructions are your own guidance to the agent, applied to every session on this machine. They land in the system prompt under a `## Standing instructions` heading, introduced as the installation operator's.

Use them for things that are true of your setup rather than of any one task:

- System policies: "Never install Python packages globally with pip. Always use `uv` or a venv."
- Installed tooling worth knowing about: "Poppler is available; use `pdftotext` for PDFs."
- Workflow preferences: "Prefer ripgrep over grep."
- Compliance rules: "Git commits on this system must be gpg-signed."

## Where they live

Instructions are content, not configuration, so they live at a conventional path beside `config.toml` rather than behind a key inside it; the one key there, [`[instructions] files`](#reading-a-projects-agentsmd), names further files and never holds text. Write:

```
~/.config/meka/instructions.md
```

If the set grows, split it into a directory instead. Every `*.md` file is concatenated in lexical order, so a numeric prefix controls the sequence:

```
~/.config/meka/instructions/
├── 00-style.md
├── 10-security.md
└── 20-tooling.md
```

The directory wins when it has content, so splitting a grown `instructions.md` is a rename rather than a migration. An empty `instructions/` falls back to the file rather than blanking your instructions. Under a custom [`MEKA_CONFIG_DIR`](../configuration/environment-variables.md), both paths follow it.

Check what is actually in effect at any time:

```bash
meka instructions show     # the resolved text, plus where it came from
meka instructions path     # the paths meka checks, and whether each exists
```

`show` prints the text on stdout and the source on stderr, so `meka instructions show 2>/dev/null` pipes cleanly.

## Passing them as a string

A file is the right shape on a workstation, but not everywhere. When the channel carrying the value is a string rather than a filesystem, use one of:

| Source | Form | |
|--------|------|---|
| `--instructions` | text | per-run, replaces the standing text |
| `MEKA_INSTRUCTIONS` | text | |
| `MEKA_INSTRUCTIONS_FILE` | path | |
| `instructions.md` / `instructions/` | file | the default |

Resolution stops at the first one set, in that order.

This matters most for containers. The [`mekabox`](https://github.com/k4yt3x/meka/blob/master/scripts/mekabox) wrapper mounts your config directory into the container **read-only** and then replaces the instructions with container-specific ones, which is a single `-e MEKA_INSTRUCTIONS=…`. Requiring a path would mean writing a temp file on the host and bind-mounting it, and the read-only mount means it could not simply write the file where meka looks.

`MEKA_INSTRUCTIONS_FILE` covers the case where a file exists but you do not control where it is mounted, such as a Kubernetes ConfigMap or a Docker secret. It accepts a directory too, since a ConfigMap mounts as a directory of keys, and in that case takes any regular file rather than only `*.md`: a ConfigMap key is often just `instructions`, and a naming choice made in someone else's YAML should not become a startup failure inside a pod.

Setting `MEKA_INSTRUCTIONS=` to the empty string means "no instructions", suppressing the file rather than falling through to it. That is the way to run a container with your host instructions mounted but not applied.

Setting both environment variables is refused at startup. There is no reading under which someone meant both, so resolving one silently would hide the mistake until the agent behaved unexpectedly.

## Reading a project's AGENTS.md

`[instructions] files` in `config.toml` names more places to read from. Each entry is a file, or a directory of files, and its text follows the standing instructions under the same heading:

```toml
[instructions]
files = ["AGENTS.md", "~/notes/meka-site.md"]
```

A relative entry is resolved against the **session's working directory**: where the REPL or a one-shot run was launched, the `cwd` an ACP client or a `POST /v1/sessions` body named, and on a resume the directory the session recorded. That is what makes `AGENTS.md` mean "this project's file" wherever the session opens. An absolute entry is read as given, and a leading `~` is expanded.

The list is empty unless you write it, and deliberately so. A relative entry lets whatever sits in the working directory speak with the operator's authority: a cloned repository's `AGENTS.md` lands in the system prompt verbatim, and under `meka serve` the client chooses the directory. Name a relative entry only where every directory a session may open in is one you trust, and prefer an absolute path for a file that is yours.

What to expect:

- An entry that is not there is skipped silently, since most directories have no `AGENTS.md`. The list has one rule, so this holds for an absolute entry too. A file that exists but cannot be read, or is not UTF-8, is skipped with a warning. Neither stops the session.
- A directory entry takes any regular file in it, in lexical order, the way `MEKA_INSTRUCTIONS_FILE` reads one.
- The files are read when a session **opens**, not at startup. A `meka -c`, a fork, a re-attach and a new `serve` session read them fresh, so an edit applies from the next open. `/cd` does not re-read them.
- `--instructions` and `MEKA_INSTRUCTIONS` replace the standing text only; the listed files still apply. Setting `MEKA_INSTRUCTIONS=` to the empty string blanks the standing text and nothing else. To stop reading a file, remove it from the list.
- Sub-agents receive them together with the standing text under `instructions: "inherit"`, and nothing otherwise.
- `GET /v1/instructions` on the HTTP API returns the standing text alone, since what a session read depends on its directory.
- The size warning below applies to the listed files as well.

`meka instructions show` and `meka instructions path` read the list against the directory they run in, so `cd` into a project and run them to see what a session opened there is told.

## When they are read

Once, at startup. Editing takes effect on the next launch, not mid-session. The files `[instructions] files` names are the exception, read when a session opens, as described above.

That is deliberate, and it follows from size. The system prompt heads the prompt-cache prefix, so a large instruction set is billed once and served from cache on every later turn. Re-reading it per turn would either invalidate that prefix whenever the file changed, or push the text down into the conversation where it would compete with actual context.

This is the opposite of [skills](./skills.md) and [memory](./memory.md), which do refresh mid-session. They can afford to: both are indexed rather than included in full, and the index is small.

`meka -c` makes restarting cheap when you do edit them.

## Notes

- Empty or whitespace-only instructions are treated as unset.
- Sub-agents do **not** receive them by default. Instructions describe the root agent, and a sub-agent handed one task by one of its turns is not that agent; inheriting the persona is how a sub-agent ends up addressing the user as though it were the one they are talking to. The agent can pass `instructions: "inherit"` to [`agent_spawn`](../tools/overview.md#agent_spawn) when a task genuinely needs the project's standing rules, or pass a [skill](./skills.md) when the direction is reusable.
- They apply at every permission level, including `none`, because you wrote them.
- A set larger than roughly 8k tokens logs a warning at startup, or when a session opens for the files `[instructions] files` names. It still works, and it is cached, but it occupies that much of every request's window and is usually a surprise rather than a decision.
- An unreadable file in a directory is skipped with a warning rather than hiding the rest of it. A path you named explicitly via `MEKA_INSTRUCTIONS_FILE` is an error instead, since running without guidance you believe you supplied is worse than not starting.
- A directory contributes at most 100 files; past that it is far more likely pointed somewhere unintended than intentional, so the rest are skipped with a warning.
