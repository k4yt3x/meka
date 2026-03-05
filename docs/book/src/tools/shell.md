# Shell Tool

## `execute_command`

Execute a shell command and return its output.

**Permission:** Read (sandboxed) / Write (unsandboxed)

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

### Read-Only Sandbox

In **read mode**, commands run inside a read-only filesystem sandbox. The child process can read files and execute programs, but any attempt to write to the filesystem is blocked by the kernel:

- **Linux**: Uses [Landlock LSM](https://landlock.io/) (kernel 5.13+). The child process is restricted via `landlock_restrict_self` before exec. Only `READ_FILE`, `READ_DIR`, and `EXECUTE` access rights are granted.
- **macOS**: Uses `sandbox-exec` with a SBPL profile that denies all `file-write*` operations.
- **Windows / unsupported platforms**: Shell commands are not available in read mode. Switch to write mode to execute commands.

In **write mode**, commands run without any sandbox restrictions.

To disable sandboxed shell execution in read mode, set `sandbox = false` under `[shell]` in the config file. When disabled, shell commands require write mode.

```toml
[shell]
sandbox = false
```

### Safety

The agent is instructed (via the system prompt) to explain what it intends to do before running potentially destructive commands. In read mode, the filesystem sandbox provides an additional layer of protection by physically preventing writes. In write mode, the permission system is the primary safety mechanism.
