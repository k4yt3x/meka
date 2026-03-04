# Shell Tool

## `execute_command`

Execute a shell command and return its output.

**Permission:** Write

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `command` | string | yes | The shell command to execute |
| `timeout_ms` | integer | no | Timeout in milliseconds (default: 30000) |

### Behavior

- Executes the command via `sh -c "<command>"` on Unix or `powershell -Command "<command>"` on Windows.
- Captures both stdout and stderr.
- Returns the exit code along with the output.
- Default timeout is 30 seconds. If the command exceeds the timeout, it is killed.
- Supports cancellation: pressing Ctrl+C while a command is running kills the child process.

### Output Format

The tool returns output in this format:

```text
Exit code: 0

stdout:
<stdout content>

stderr:
<stderr content>
```

### Examples

```text
agsh [w] > run the test suite
```

The agent will call `execute_command` with `command: "cargo test"`.

```text
agsh [w] > show disk usage of the current directory
```

```text
agsh [w] > check which ports are listening on this machine
```

```text
agsh [w] > compile the project in release mode
```

### Timeout

For long-running commands, the agent can specify a custom timeout:

```text
agsh [w] > run the full integration test suite (it might take a while)
```

The agent may call `execute_command` with a higher `timeout_ms` value.

### Safety

The agent is instructed (via the system prompt) to explain what it intends to do before running potentially destructive commands. However, the permission system is the primary safety mechanism: write-permission tools are only available when you explicitly enable write mode.
