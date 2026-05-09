# Sessions

Sessions persist your conversation so you can resume later. Each session is identified by a UUID and stored in a SQLite database.

## How Sessions Work

- A session is **not** created when agsh starts. It is created lazily when you send the first message.
- When a session is created, its UUID is printed to stderr.
- When you exit agsh (Ctrl+D), the session UUID is printed again so you can note it for later.
- Sessions include the full conversation: your inputs, the agent's responses, and tool call results.

## Resuming a Session

### Continue Last Session

```bash
agsh -c
```

This resumes the most recently updated session.

### By UUID

```bash
agsh -c 550e8400-e29b-41d4-a716-446655440000
```

The agent loads the previous conversation and continues from where you left off.

### By UUID Prefix

If the value isn't a valid UUID, agsh treats it as a leading prefix and looks up sessions whose ID starts with it. This avoids having to copy the entire UUID:

```bash
agsh -c 550e            # works if exactly one session starts with `550e`
agsh -c 5               # likely ambiguous; agsh lists matching IDs and exits
```

When a prefix matches multiple sessions, agsh prints the matching IDs (most-recent first) so you can disambiguate. Type a few more characters until the prefix is unique.

## Session Locking

Only one agsh instance can be attached to a session at a time. This prevents race conditions from concurrent writes.

- If you try to resume a session that is locked by a running agsh process, you will get an error.
- If the locking process has exited (crashed or was killed), agsh detects this and allows you to take over the lock.

## Storage Location

Sessions are stored in a SQLite database at a platform-specific location:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/agsh/sessions.db` (`$XDG_DATA_HOME/agsh/sessions.db`) |
| macOS | `~/Library/Application Support/agsh/sessions.db` |
| Windows | `%APPDATA%\agsh\sessions.db` |

## Database Schema

The database has three tables:

**sessions** -- one row per session:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Primary key |
| `created_at` | TEXT (RFC 3339) | When the session was created |
| `updated_at` | TEXT (RFC 3339) | When the session was last updated |
| `locked_by` | TEXT (PID) | PID of the process holding the lock, or NULL |
| `metadata` | TEXT | Reserved for future use |

**messages** -- one row per message in a session:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing primary key |
| `session_id` | TEXT (UUID) | Foreign key to `sessions.id` |
| `role` | TEXT | `user`, `assistant`, or `tool_results` |
| `content` | TEXT | Message content (plain text or JSON) |
| `created_at` | TEXT (RFC 3339) | When the message was saved |

**tool_outputs** -- scratchpad entries, one row per entry:

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT (UUID) | Part of composite primary key |
| `name` | TEXT | Part of composite primary key |
| `content` | TEXT | The stored content |
| `created_at` | TEXT (RFC 3339) | When the entry was created |

Scratchpad entries are scoped to a session. Two sessions can have entries with the same name. Entries are preserved across compaction but deleted when a session is deleted.

## History Retention

agsh automatically manages session storage on startup with sensible defaults:

- **`retention_days`** (default: `90`) -- deletes sessions whose `updated_at` is older than this many days.
- **`max_storage_bytes`** (default: `52428800` / 50 MB) -- when total message content exceeds this limit, the oldest sessions are deleted until the total is under the limit.

You can override these defaults in the config file under `[session]`:

```toml
[session]
retention_days = 30          # delete sessions not used in 30 days
max_storage_bytes = 10485760 # cap total storage at ~10 MB
```

See [Config File](../configuration/config-file.md#session) for details.

## Context Window Limiting

Long sessions can exceed the LLM's context window or become expensive. The `context_messages` setting (default: `200`) limits how many recent messages are sent to the API:

```toml
[session]
context_messages = 100
```

The full history remains in SQLite for resumption. Only the API payload is truncated. The truncation preserves tool call chains (it never splits a tool use from its result).

### Compacting a Session

If a session becomes too long, you can use the `/compact` command to have the LLM summarize the conversation and replace older messages with a structured summary. Recent messages are preserved verbatim. The summary includes key files, decisions, errors, and user preferences.

Compaction preserves scratchpad entries and the todo list, and re-injects environment context so the agent isn't disoriented after compaction. Tools that the model loaded via `load_tool` before compaction stay loaded after — the deferred-tool active set is snapshotted into the compaction boundary, so resumed sessions don't re-issue `load_tool` for tools they already used.

Internally, compaction does not delete pre-compaction rows from the database. It appends a `compact_boundary` row to the `messages` table; the materialized view is reconstructed from the event log, so the persisted log itself stays append-only.

### Auto-Compact

When `auto_compact` is enabled (default: `true`), agsh automatically compacts the conversation when the input token count exceeds 80% of the context window. This runs between turns, not during tool loops.

```toml
[session]
auto_compact = true
context_window = 200000  # optional override
```

## Listing Sessions

To see past sessions:

```bash
agsh list
```

This shows a table with each session's ID, last update time, and a preview of the first message:

```
ID                                    Updated              Preview
550e8400-e29b-41d4-a716-446655440000  2026-03-14 12:00:00  How do I implement a binary search tree?
a1b2c3d4-e5f6-7890-abcd-ef1234567890  2026-03-13 09:30:00  Fix the login page CSS
```

By default the 20 most recent sessions are shown. Use `-n` to change:

```bash
agsh list -n 50
```

## Exporting a Session

You can export any session as a Markdown file:

```bash
agsh export 550e8400-e29b-41d4-a716-446655440000
```

This writes `session-550e8400-e29b-41d4-a716-446655440000.md` in the current directory with the full conversation. User and assistant messages are rendered as Markdown sections, while tool calls and results are wrapped in collapsible `<details>` blocks.

To write to a specific file:

```bash
agsh export 550e8400-e29b-41d4-a716-446655440000 -o conversation.md
```

To print to stdout (for piping):

```bash
agsh export 550e8400-e29b-41d4-a716-446655440000 -o -
```

## Deleting Sessions

Delete specific sessions by UUID:

```bash
agsh delete 550e8400-e29b-41d4-a716-446655440000
```

Delete multiple sessions at once:

```bash
agsh delete 550e8400-e29b-41d4-a716-446655440000 a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

Delete all sessions:

```bash
agsh delete --all
```

## Managing Sessions via SQLite

You can also manage sessions directly through the SQLite database. For example, to list all sessions:

```bash
sqlite3 ~/.local/share/agsh/sessions.db \
  "SELECT id, created_at, updated_at FROM sessions ORDER BY updated_at DESC;"
```
