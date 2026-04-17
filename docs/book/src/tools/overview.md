# Tools Overview

Tools are the actions that the agent can perform on your behalf. The LLM decides which tools to call based on your instructions.

## Available Tools

| Tool | Permission | Description |
|------|-----------|-------------|
| [`read_file`](./file-operations.md#read_file) | Read | Read file contents |
| [`edit_file`](./file-operations.md#edit_file) | Write | Make string replacements in a file |
| [`write_file`](./file-operations.md#write_file) | Write | Create or overwrite a file |
| [`find_files`](./search.md#find_files) | Read | Find files by glob pattern |
| [`search_contents`](./search.md#search_contents) | Read | Search file contents with regex |
| [`fetch_url`](./web.md#fetch_url) | Read | Fetch a web page as markdown |
| [`web_search`](./web.md#web_search) | Read | Search the web |
| [`execute_command`](./shell.md#execute_command) | Read/Write | Run a shell command |
| [`todo_write`](./overview.md#todo_write) | Read | Manage a structured task list |
| [`spawn_agent`](./overview.md#spawn_agent) | Read | Delegate tasks to a sub-agent |
| [`scratchpad_write`](./scratchpad.md#scratchpad_write) | Read | Store content in the scratchpad |
| [`scratchpad_read`](./scratchpad.md#scratchpad_read) | Read | Read a scratchpad entry |
| [`scratchpad_edit`](./scratchpad.md#scratchpad_edit) | Read | Edit a scratchpad entry |
| [`scratchpad_list`](./scratchpad.md#scratchpad_list) | Read | List scratchpad entries |
| [`scratchpad_delete`](./scratchpad.md#scratchpad_delete) | Read | Delete a scratchpad entry |
| [`skill`](./overview.md#skill) | Read | Load a named skill's instructions |
| [`render_image`](./overview.md#render_image) | Read | View an image from in-memory base64 or scratchpad |

## Permission Requirements

Tools are grouped by the minimum permission level required:

**Read permission** (available in read, ask, and write modes):
- `read_file`, `find_files`, `search_contents`, `fetch_url`, `web_search`
- `execute_command` (sandboxed, filesystem write-protected)
- `todo_write`, `spawn_agent`, `skill`, `render_image`
- All scratchpad tools

**Write permission** (only available in write mode):
- `edit_file`, `write_file`, `execute_command` (unsandboxed)

In **ask** mode, all tools are available but each call requires user confirmation.

In **none** mode, no tools are available. The agent can only respond with text.

## Scratchpad Parameter

All tools support an optional `scratchpad` string parameter. When provided, the tool's output is saved to the scratchpad under that name instead of being returned inline. This lets the agent store large outputs for later processing without consuming conversation context.

```text
execute_command({"command": "pdftotext doc.pdf -", "scratchpad": "pdf_text"})
```

## How Tool Calls Work

1. The agent receives your instruction and decides which tools to call
2. For each tool call, agsh checks the current permission level
3. In ask mode, you are prompted to approve or deny each tool call
4. If permitted, the tool executes and its output is fed back to the agent
5. The agent may make additional tool calls or respond with text
6. This loop continues until the agent has no more tool calls to make

Tool calls and their results are displayed in the terminal so you can see what the agent is doing.

## `todo_write`

A built-in tool for managing a structured task list during a session. The agent uses this to track multi-step work and communicate progress. The task list is displayed in the terminal and injected into the conversation context each turn.

## `spawn_agent`

Spawns a read-only sub-agent to perform research or analysis tasks. The sub-agent has access to the same tools (except `spawn_agent` and `todo_write`) and returns a report. This is useful for delegating exploration without polluting the main conversation context.

## `skill`

Loads a named skill's instructions. Skills are user-defined knowledge packages stored in `~/.config/agsh/skills/<name>/SKILL.md`. The system prompt lists available skills with their description and when-to-use hint; the agent calls `skill({"name": "<skill-name>"})` to load the full body. See [Skills](../usage/skills.md) for how to author skills.

## `render_image`

Displays an image the agent has in memory — as base64 bytes or in a scratchpad entry — as a multimodal content block. Complements `fetch_url` (network) and `read_file` (local file) by covering the third case: image data produced on the fly by a command pipeline.

Typical workflow:

```text
execute_command({"command": "ffmpeg -i input.mp4 -vframes 1 -f image2pipe pipe: | base64 -w0", "scratchpad": "frame"})
render_image({"from_scratchpad": "frame"})
```

Parameters:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `from_scratchpad` | string | one of two | Name of a scratchpad entry containing base64-encoded image bytes |
| `base64` | string | one of two | Base64-encoded image bytes, passed inline |

Exactly one of `from_scratchpad` or `base64` must be provided. Prefer `from_scratchpad` for large images — inline base64 inflates tool-call JSON.

The bytes must decode to a supported raster image. PNG, JPEG, GIF, WebP, and BMP pass through unchanged; TIFF, ICO, HDR, EXR, TGA, PNM, QOI, DDS, and Farbfeld are auto-converted to PNG. Size cap is ~3.75 MB on the final payload.

Only call `render_image` when the current model supports vision input.

## Redirecting output to the scratchpad

Several tools — `execute_command`, `find_files`, `search_contents`, `fetch_url`, `spawn_agent` — accept an optional `scratchpad` parameter that redirects their output to a named scratchpad entry instead of returning it inline. When this parameter is set, the tool produces its **full, untruncated output**: internal result-count caps (`find_files` 200, `search_contents` 100) and length caps (`fetch_url` `max_length`) are lifted for the scratchpad-bound result.
