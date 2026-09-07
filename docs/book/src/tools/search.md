# Search tools

Both tools default to sweeping every [workspace root](../usage/acp.md#multi-root-workspaces): the
working directory, plus any extra folders an ACP client supplied. Passing `path` searches exactly
that tree instead.

## `find_files`

Find files matching a glob pattern.

**Permission:** Read

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `glob` | string | yes | Glob pattern to match files against |
| `path` | string | no | Directory to search in. Omitted, every workspace root is walked |
| `limit` | integer | no | Maximum results to return, at least 1 (defaults to 500 inline) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Results are limited to 500 matches inline; `limit` raises the cap and `scratchpad` lifts it.
- Returns one file path per line.
- The walk stops after 60 seconds. The result set is still returned, with a note saying it is
  incomplete, so a search rooted too high in the tree costs a minute rather than hanging the turn.
- Interrupting the turn (Ctrl+C, or `session/cancel` from an editor) stops the walk.
- Paths that cannot be read are skipped and counted; the total is reported once at the end rather
  than logged per path.

### Glob patterns

| Pattern | Matches |
|---------|---------|
| `*.rs` | All `.rs` files in the current directory |
| `**/*.rs` | All `.rs` files recursively |
| `src/*.txt` | All `.txt` files in `src/` |
| `test_*` | All files starting with `test_` |

---

## `search_contents`

Search file contents using a regex pattern. Powered by the ripgrep library.

**Permission:** Read

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `pattern` | string | yes | Regex pattern to search for |
| `path` | string | no | File or directory to search in. Omitted, every workspace root is walked |
| `glob` | string | no | Glob pattern to filter which files are searched (e.g., `*.rs`) |
| `limit` | integer | no | Maximum matches to return, 1 to 100 (default: 100; unbounded with `scratchpad` unless set) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Searches recursively through directories.
- Skips hidden files (starting with `.`), the `target` and `node_modules` directories, and, below
  `unrestricted`, meka's own private directories: the config directory, the data directory holding
  `meka.db`, and the command-output captures. `find_files` steps around the same three.
- **`.gitignore` is not honored.** Only the matcher comes from ripgrep; the walk is meka's own, and
  those four exclusions are all of it. A build directory that is ignored but not named above is
  searched, so pass `glob` or `path` to stay out of one.
- Results are limited to 100 matches; `limit` lowers the cap, and `scratchpad` lifts it unless
  `limit` is also set. The search stops once the cap is exceeded instead of reading the rest of
  the tree to fill a result set it will truncate anyway.
- The search stops after 60 seconds, returning what it found with a note saying it is incomplete.
- Interrupting the turn (Ctrl+C, or `session/cancel` from an editor) stops the search.
- Each result includes the file path, line number, and matching line.
