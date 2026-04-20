# Shell Tool

## `execute_command`

Execute a shell command and return its output.

**Permission:** Read (sandboxed) / Write (unsandboxed)

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `command` | string | yes | The shell command to execute |
| `timeout_ms` | integer | no | Timeout in milliseconds (default: 30000) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Executes the command via `sh -c "<command>"` on Unix, or `powershell.exe -NoProfile -NonInteractive -Command "<command>"` on Windows (same shell in both sandboxed and unsandboxed mode).
- Captures both stdout and stderr.
- Returns the exit code along with the output if non-zero.
- Oversized output is losslessly persisted to the scratchpad by the agent layer — the tool itself never truncates.
- Default timeout is 30 seconds. If the command exceeds the timeout, it is killed (on Unix, via the process group so backgrounded grandchildren are caught too).
- Supports cancellation: pressing Ctrl+C while a command is running kills the child process.

### Shell-specific semantics

- **Unix (`sh -c`)**: POSIX `$VAR` expansion applies. Pass a literal `$` with single quotes (`'$foo'`) or backslash escape (`\$foo`).
- **Windows (`powershell.exe -Command`)**: The script body reaches PowerShell directly. Use PowerShell syntax (`$var = ...`, `$env:PATH`) — and crucially, **do not** wrap your command in another `powershell -Command "..."`. The outer PowerShell will expand your inner `$var` references to empty strings before the inner shell runs, producing a parser error on mangled syntax. If you need to invoke a nested script, drop it into a `.ps1` file and run it by path, use `-EncodedCommand <base64>`, or escape each `$` as `` `$ ``.

### Read-Only Sandbox

In **read mode**, commands run inside a filesystem sandbox that blocks writes to the user's real data. Reads and program execution still work normally:

- **Linux**: Uses [Landlock LSM](https://landlock.io/) (kernel 5.13+). The child process is restricted via `landlock_restrict_self` before exec. Only `READ_FILE`, `READ_DIR`, and `EXECUTE` access rights are granted — writes anywhere on the filesystem return `EACCES`.
- **macOS**: Uses `sandbox-exec` with a SBPL profile that denies all `file-write*` operations.
- **Windows**: Spawns the child with a duplicated primary token dropped to **Low integrity** (`SECURITY_MANDATORY_LOW_RID`) via `SetTokenInformation(TokenIntegrityLevel, …)`. Writes to the home directory, `%APPDATA%`, Program Files, and system directories — any location with Medium-or-higher integrity ACLs — are blocked by the kernel. Unlike Landlock, Low integrity is not a total write-denial: the child can still write to the small residual Low-integrity-writable surface (`%LOCALAPPDATA%\Low`, `%TEMP%\Low`, any path with an explicit Low-integrity write ACE) and to files it creates itself (which inherit Low integrity). For practical purposes this matches the guarantees of `sandbox-exec` and prevents the agent from touching user data, but full "zero writes anywhere" on Windows would require Windows Sandbox or an AppContainer and is out of scope.
- **Unsupported platforms**: Shell commands are not available in read mode — switch to write mode to execute commands without a sandbox.

In **write mode**, commands run without any sandbox restrictions.

To disable sandboxed shell execution in read mode, set `sandbox = false` under `[shell]` in the config file. When disabled, shell commands require write mode.

```toml
[shell]
sandbox = false
```
