# Shell tool

## `shell_execute`

Execute a shell command and return its output.

**Permission:** `read` (sandboxed read-only) / `workspace` (sandboxed, writable inside the workspace) / `unrestricted` (unsandboxed)

### Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `command` | string | yes | The shell command to execute |
| `timeout_ms` | integer | no | Timeout in milliseconds (default: 30000) |
| `scratchpad` | string | no | Save output to the scratchpad under this name |

### Behavior

- Executes the command via `sh -c "<command>"` on Unix, or `powershell.exe -NoProfile -NonInteractive -Command "<command>"` on Windows, whether or not the command is sandboxed.
- Captures both stdout and stderr.
- Returns the exit code along with the output if non-zero.
- Oversized output is losslessly persisted to the scratchpad by the agent layer; the tool never truncates what it returns to the agent.
- A command may print up to 64 MiB across stdout and stderr. Past that it is stopped, the way a timeout stops it, and the result states the reason and then carries everything it printed up to the stop. Nothing is elided and nothing is written to disk: the whole output is in the result, and in the scratchpad when it is too large for the conversation. The bound exists because a command that writes faster than the turn ends (`cat /dev/zero`, a runaway build log) would otherwise grow one buffer until the process died; a command that hits it wants a narrower rerun (`grep`, `head`, `tail`), not a larger bound.
- Default timeout is 30 seconds. If the command exceeds the timeout, it is killed with every process it started (on Unix through its process group, on Windows through its Job Object), and the result keeps what it printed before the kill.
- Supports cancellation: pressing Ctrl+C while a command is running kills the command and everything it started.

### Shell-specific semantics

- **Unix (`sh -c`)**: POSIX `$VAR` expansion applies. Pass a literal `$` with single quotes (`'$foo'`) or backslash escape (`\$foo`).
- **Windows (`powershell.exe -Command`)**: The script body reaches PowerShell directly. Use PowerShell syntax (`$var = ...`, `$env:PATH`), and crucially, **do not** wrap your command in another `powershell -Command "..."`. The outer PowerShell will expand your inner `$var` references to empty strings before the inner shell runs, producing a parser error on mangled syntax. If you need to invoke a nested script, drop it into a `.ps1` file and run it by path, use `-EncodedCommand <base64>`, or escape each `$` as `` `$ ``.

### Network

The sandbox leaves the network open by default: the threat model is "no state mutation", and `curl http://x | pdftotext` must keep working. [`[permissions].network`](../configuration/config-file.md#permissions) narrows it for a confined command and for `web_fetch` alike. `none` cuts a command off: Bubblewrap unshares the network namespace, so not even loopback answers; Landlock handles every TCP bind and connect from ABI v4 and grants no port, which leaves UDP open since no Landlock ABI has a right over it; `sandbox-exec` withholds its outbound rule under `(deny default)`, which closes every socket while the system resolver's Mach service stays reachable, so a command can still look names up; the Windows AppContainer grants no network capability. `remote` keeps a command off this machine while other machines stay reachable, which the AppContainer holds; Bubblewrap, Landlock and `sandbox-exec` hold it as `none`, cutting the command off whole, and startup says so once with the value to set that silences the warning. The Windows restricted token cannot cut a command off at all and warns once at startup. The setting reaches `web_fetch` too, which runs in the meka process where no sandbox reaches: withheld whole under `none`, and under `remote` refusing this machine's own addresses at the URL, at every redirect and at the resolver; with a proxy configured the proxy resolves the target and the resolver sees the proxy's name alone, as under `[web].refuse_private_destinations`. It never reaches an MCP server, which is a process outside the sandbox: withhold the server instead. A sub-agent holds its parent's setting unless its spawn narrows it with `network`, and the scope lines of a session whose network is narrowed name what the shell and `web_fetch` each reach, `Network: none (shell and web_fetch)` or `Network: remote (web_fetch), none (shell)`.

### Memory

A command that takes every byte the machine has is the process that loses it, not meka, wherever the platform has a lever, so the session survives and the result says the command died:

- **Linux**: every process meka starts runs at the highest `oom_score_adj`, inherited by the shell, the Bubblewrap or Landlock layer around it and everything they start. Whether the service's cgroup or the whole host runs out of memory, the kernel's OOM killer takes that process, a stdio MCP server included, ahead of meka and of everything else on the host, whatever meka's own score is.
- **Windows**: every process meka starts runs in a Job Object whose committed memory stops at the machine's physical memory. Past it the command's own allocations fail and it exits; meka's never do. The cap is per command, so two commands can still exhaust the machine together. The job ends with the command: when it returns, times out or is stopped, every process it started is ended, so a server a command left running in the background does not outlive the call. A program started through an app execution alias (the `python` or `winget` shims under `WindowsApps`) runs outside the command's job, so neither the cap nor the kill reaches it; name such an executable by its real path.
- **macOS**: no lever exists for one process over another. The system takes the largest footprint first, which is the runaway command.

This covers every process meka starts: shell commands, the screen's `!` commands, schedule gates, stdio MCP servers and an HTTP MCP server's `headers_helper`. Two exceptions are the user's own processes: the editor `meka skill edit` and `meka memory edit` open, and the browser the screen opens for a URL a server asks for.

Running `meka serve` under systemd, give the unit a ceiling and tell systemd that a command's death is not meka's:

```ini
[Service]
MemoryMax=75%
MemorySwapMax=0
OOMPolicy=continue
OOMScoreAdjust=-900
```

`MemoryMax=75%` is relative to the machine's physical memory, so the unit moves with its host. `OOMPolicy=continue` keeps the unit running when the kernel kills a command inside it; the default, `stop`, ends meka too. `OOMScoreAdjust=-900` ranks meka below everything else the host runs.

### Read-only sandbox

At **`read`**, commands run inside a sandbox that blocks writes to the user's real data. Reads, program execution, and network access still work normally: the threat model is "no state mutation, but `curl http://x | pdftotext` must keep working."

#### What's blocked vs allowed (across all backends)

| Surface | Blocked | Allowed |
|---|---|---|
| Filesystem writes outside tmp / Low-integrity paths | ✓ | |
| File metadata changes (mode, owner, timestamps, extended attributes) | Bubblewrap / macOS / Windows restricted token / Windows AppContainer, which alone still lets a command rewrite the ACL of a file the user owns, an owner right no grant withdraws | Landlock alone, on every kernel |
| Filesystem reads of meka's config directory and store | Bubblewrap / Landlock / macOS / Windows AppContainer | Windows restricted token |
| Filesystem reads outside the session's roots and the system directories, with [`[permissions].confine_reads`](../usage/permissions.md#the-read-boundary) on | Bubblewrap / Landlock / macOS / Windows AppContainer | Windows restricted token |
| Filesystem reads elsewhere, with `confine_reads` off | | ✓ |
| Unix sockets the Bubblewrap masks do not cover (`$HOME`, `/var/lib`, the abstract namespace) | Landlock, and Bubblewrap on kernel 7.1+ (6.12+ for the abstract namespace) | Bubblewrap on older kernels / Windows |
| Program execution | | ✓ |
| Outbound network, with [`[permissions].network`](../configuration/config-file.md#permissions) at its default `host` | | ✓ |
| Outbound network, with `[permissions].network = "none"` | Bubblewrap (the namespace is unshared) / Landlock (every TCP port, from ABI v4; UDP stays open) / macOS / Windows AppContainer | Windows restricted token |
| Outbound network to this machine, with `[permissions].network = "remote"` (other machines stay reachable) | Windows AppContainer; Bubblewrap / Landlock / macOS hold it as `none` and cut the command off whole | Windows restricted token |
| dbus / systemd-user state mutations | Bubblewrap / macOS / Landlock | Windows |
| Mach IPC state mutation (launchd, pasteboard, LaunchServices) | macOS | Linux / Windows |
| The user's clipboard, other windows' handles and messages, input injection, global atoms, system parameters, display settings and desktop switches (Windows) | Windows, on both backends: the job's UI restrictions, with the IME and injection bits from the builds that have them | |
| COM / RPC to Low-integrity-accepting services (Windows) | | ✓ |
| Inheritance of sensitive parent env vars (API keys, OAuth tokens, …) | ✓ (all platforms) | |

The sandbox is not an adversarial containment boundary; it's defense-in-depth against an agent accidentally modifying user data. Set permission to `none` if you don't trust a turn at all.

#### Scratch space

Whether a confined command has somewhere to write temporary files differs by backend:

| Backend | Scratch space | Effect |
|---|---|---|
| Bubblewrap (`read`) | Private `/tmp` tmpfs, gone with the sandbox | `mktemp`, Python's `tempfile`, `gcc`, `patch` and `pip` builds all work |
| Landlock (`read` and `workspace`) | None outside the roots | Those tools fail with `Permission denied` |
| Windows `workspace` | None outside the roots | `New-TemporaryFile` is denied (measured) |
| macOS Seatbelt (`read`) | Per-backend; see below | |

Under Bubblewrap the child writes into an in-memory `/tmp` that exists only inside the sandbox's mount namespace, so nothing real is touched. Landlock can only decide which real paths a process may touch, and the only temporary directory meka could offer there would be a real one under the real `/tmp`; below `unrestricted` meka writes to nothing but its own store, so it offers none. A tool that needs temp files under Landlock is the reason to install Bubblewrap. `git` needs none of this: its lock file lives inside the repository, which is read-only under both backends alike.

The practical cost is diagnostic: the failure is a bare `Permission denied` naming a path in `/tmp` (or `%TEMP%` on Windows), with nothing in the message connecting it to the sandbox. If a command fails that way and you expected it to work, add the directory it wants as a writable root at `workspace`.

#### Environment variable scrubbing

The `read` sandbox still permits outbound network (the threat model intentionally keeps `curl http://x | pdftotext`-style pipelines working), so any secret in the parent process's environment (`ANTHROPIC_API_KEY`, `AWS_SECRET_ACCESS_KEY`, `GITHUB_TOKEN`, OAuth tokens, etc.) would be a live exfiltration vector under prompt injection. meka scrubs the child environment at spawn time across every backend (Bubblewrap, Landlock, Seatbelt, and both Windows backends).

- **Unix (Linux + macOS): allow-list.** Only a curated set of vars survives into the `read` child: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `PWD`, `TERM`, `COLORTERM`, `LANG`, `TMPDIR`, `TMP`, `TEMP`, plus everything matching the `LC_*` and `XDG_*` prefixes. Because `read` intentionally keeps outbound network working, the proxy and CA-bundle vars survive too: `HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`, `ALL_PROXY` (and their lowercase spellings), `SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` and `NODE_EXTRA_CA_CERTS`, several of which redirect TLS trust, so treat them as part of the boundary. Anything else is dropped, including credential-shaped vars (`AWS_*`, `GITHUB_TOKEN`, `OPENAI_API_KEY`, …) and credential-pointer vars (`SSH_AUTH_SOCK`, `KUBECONFIG`, `GNUPGHOME`, `NETRC`, `GIT_ASKPASS`, `GIT_SSH_COMMAND`, etc.) as well as benign-but-unlisted vars like `EDITOR`, `PAGER`, `DISPLAY`, custom toolchain vars, and so on. Unknown vars are dropped by default.
- **Windows: deny-list.** PowerShell pulls in a long tail of system vars (`PSModulePath`, `APPDATA`, `ProgramFiles`, etc.) that don't fit a tidy allow-list, so the Windows path lets everything through *except* names that match a heuristic deny-list. Dropped names include:
    - Credential-shaped substrings: `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, `*PASSPHRASE*`, `*API_KEY*`, `*_KEY*`, `*BEARER*`, `*CREDENTIAL*`, etc.
    - Credential-pointer substrings: `SSH_AUTH_SOCK`, `KUBECONFIG`, `GNUPGHOME`, `NETRC`, `GIT_ASKPASS`, `SSH_ASKPASS`, `GIT_SSH_COMMAND`.
    - Provider / service prefixes: `ANTHROPIC_*`, `OPENAI_*`, `AWS_*`, `GCP_*`, `GOOGLE_*`, `AZURE_*`, `GITHUB_*`, `OPENROUTER_*`, `GROQ_*`, `MISTRAL_*`, `COHERE_*`, `DATABASE_*`, `POSTGRES_*`, `MONGO_*`, `STRIPE_*`, `CLOUDFLARE_*`, `VAULT_*`, `OAUTH_*`, `JWT_*`, `SENTRY_*`, `SLACK_*`, `DISCORD_*`, and others; see `is_sensitive_env_name` in `src/sandbox.rs` for the full list.

  The deny-list is intentionally aggressive on false positives (a legitimate `GITHUB_ACTOR` is dropped alongside `GITHUB_TOKEN`) because the cost of a missing env var is a confusing tool error, while the cost of a leaked credential is a live exfiltration channel.

**`unrestricted` keeps the full parent environment.** This is the trusted-operation path where users legitimately need `NPM_TOKEN` for `npm publish`, `AWS_*` creds for `aws s3 cp`, `GH_TOKEN` for `gh pr create`, etc. If you need a specific var inside a sandboxed shell command, switch to it for that turn.

An approved command is not an exception: with [approvals](../usage/permissions.md#approvals) on, a command you approve at `read` or `workspace` still runs in that level's sandbox with the scrubbed environment. Approval never widens reach, so the prompt is a question about whether to run the command, not a grant of the environment behind it.

#### Linux: pick a backend

Linux has two sandbox mechanisms and three values for `[shell].sandbox_backend` in `config.toml`:

- **Bubblewrap** (`sandbox_backend = "bubblewrap"`, recommended): wraps the command in `bwrap` with `--ro-bind /`, tmpfs masks over `/run`, `/tmp`, `/var/tmp`, and `$XDG_RUNTIME_DIR`, plus `--unshare-user --unshare-pid --unshare-uts --unshare-ipc`. The tmpfs masks make the dbus session bus, systemd-user socket, and other socket-on-disk IPC paths unreachable, so `systemctl --user start <unit>`, `dbus-send`, and similar state-changing calls fail. Network is not unshared. On a kernel with Landlock ABI v6 (6.12) or newer, the command also runs inside meka's Landlock ruleset: bwrap execs `meka confine`, which enacts the ruleset after the mounts and then runs the shell. That closes what the masks leave open: the abstract socket namespace and signals to outside processes, ioctls on real devices, and from kernel 7.1 a socket under `$HOME` or `/var/lib` too. Requires the `bubblewrap` package and a kernel with user-namespace creation enabled; Ubuntu 24.04 also needs the `bwrap-userns-restrict` AppArmor profile from `apparmor-profiles`.
- **Bubblewrap with Landlock** (`sandbox_backend = "bubblewrap-landlock"`): the same as Bubblewrap, but the Landlock layer inside is required rather than added where the kernel allows. On a kernel below 6.12 the backend is reported unusable and the shell at `read` fails instead of running under bwrap alone. What the layer closes still follows the kernel, sockets on disk outside the masks only from 7.1. Auto-detection never picks it; pin it on a host where it must be certain the layer is there.
- **Landlock** (`sandbox_backend = "landlock"`, fallback): uses the [Landlock LSM](https://landlock.io/). Blocks filesystem writes via `landlock_restrict_self`. **Requires ABI v9 (kernel 7.1+)**: that is the first ABI with a right over `connect()` to Unix sockets on disk, and below it a sandboxed shell can invoke state-mutating dbus methods and `systemd-run --user` escapes the filesystem restriction entirely. meka reports Landlock unusable on older kernels rather than sandboxing with a ruleset that leaves that door open, which means kernels below 7.1 (Debian 13, RHEL 10, Ubuntu 24.04 and 26.04 LTS) need Bubblewrap installed for the shell at `read`. Closing that route also breaks socket-based clients such as `docker` and `psql` at `read`. meka's own directories are hidden by never being covered by a rule: reading files is granted per sibling along the path to each of them, so their names and sizes stay visible but their bytes do not, and the one cost is that a workspace root above them (a root at `$HOME`) can create new files only in its subdirectories. What no Landlock ABI mediates is file metadata: a command at `read` can still `chmod`, `chown`, `touch` and set extended attributes on any file you own. Prefer Bubblewrap, whose read-only bind refuses those and which removes the socket-on-disk paths on any kernel.

`sandbox_backend` is unset unless you pin it yourself; no command writes it. When unset, meka probes Bubblewrap once at startup and prefers it when available, falling back to Landlock with one warning at startup: that it did so, and that Landlock cannot stop a command changing a file's mode, owner, timestamps or attributes. Pinning `sandbox_backend = "landlock"` accepts that and silences the warning.

```toml
[shell]
sandbox = true                       # default; set to false to disable
sandbox_backend = "bubblewrap"       # or "landlock" or "bubblewrap-landlock"; unset = auto-detect
```

#### macOS

Uses `sandbox-exec` with a hardened SBPL profile (modeled after [Codex](https://github.com/openai/codex)'s vendored seatbelt policy, which is itself based on Chrome's renderer sandbox). The profile is closed-by-default: filesystem writes are blocked, Mach-lookup is restricted to a curated allow-list of safe services, and mutation paths (launchd job control, pasteboard, LaunchServices, distributed notifications) are not in the allow-list. Network and DNS resolution remain available. macOS has one backend, so `sandbox_backend` takes no value there; a value from another platform is refused.

#### Windows: pick a backend

Windows has two backends and two values for `[shell].sandbox_backend`:

- **AppContainer** (`sandbox_backend = "appcontainer"`, preferred): runs the command inside a process security environment the OS enforces, on a Windows 11 build that has the API: 26H2, where meka measured it, and 24H2 and 25H2 from the August 2026 update, where Microsoft lists it. meka's config directory and store are denied, the read boundary and a sub-agent's scope hold for the shell as they do for the file tools, `network = "none"` cuts a command off, and no ACE is written to your directories.
- **Restricted token** (`sandbox_backend = "restricted-token"`, fallback): spawns the child with a duplicated primary token dropped to **Low integrity** (`SECURITY_MANDATORY_LOW_RID`) via `SetTokenInformation(TokenIntegrityLevel, …)` at `read`, and under a `WRITE_RESTRICTED` token plus an ACE per workspace directory at `workspace`. Writes to the home directory, `%APPDATA%`, Program Files, and system directories (any location with Medium-or-higher integrity ACLs) are blocked by the kernel. Low integrity also strips token privileges, and the same env scrubbing applied on Unix runs here (see [Environment variable scrubbing](#environment-variable-scrubbing) above). It confines writes only: a command can read meka's store, `confine_reads` holds for the file tools alone, and `network = "none"` for `web_fetch` alone.

`sandbox_backend` is unset unless you pin it yourself; no command writes it. When unset, meka probes AppContainer once at startup and prefers it when the build has the API at schema 1.1, falling back to the restricted token with one warning at startup naming what it gives up. A build whose API speaks schema 1.0 has no ingress table and so no loopback exemption: it is not auto-picked, the warning says so, and pinning `sandbox_backend = "appcontainer"` takes it with `network = "host"` held as `remote`. Pinning `sandbox_backend = "restricted-token"` accepts the fallback and silences the warning. A value from another platform (`bubblewrap`, `landlock`) is refused: the shell at `read` fails and startup names the values Windows takes.

Under AppContainer, `read` grants every drive root read-only and denies meka's config and data directories, `workspace` adds a read-write grant per workspace directory, and `confine_reads` or a sub-agent's scope replaces the drive roots with the readable paths; the environment is created per command and closed after it, which also removes the private `%TEMP%` the OS gives the command. Measured limits of the mechanism, not settings. A read-only grant refuses the directory open that `cd` makes, so `cd` from `cmd`, Python's `os.chdir` and git's own directory change (`git -C`, hooks) fail into a read-only directory, while the command's initial directory, PowerShell's `Set-Location` and `cd` into a workspace directory work; read-only git inspection of a checkout outside the workspace therefore fails at `read` and works at `workspace`. msys2 `bash.exe` cannot start at all, so a tool that runs git's shell scripts fails. A volume without NTFS permissions (a FAT32 removable drive, measured) is unreadable, since it carries no ACE the environment could honor. An app execution alias, such as the `python.exe` the Python install manager places under `WindowsApps`, fails to start with "The file cannot be accessed by the system", while the real executable runs by its own path. A mapped network drive is out of reach, and while one is mapped PowerShell prints `Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.` on stderr once per command, which is noise and not a failure. Pin `sandbox_backend = "restricted-token"` where any of these matters more than the boundary. A meka started from an SSH session or a service grants its own window station to AppContainer processes at startup (never the interactive `WinSta0`, which has the grant already), since PowerShell cannot start inside the environment without it; a failed grant is warned about once.

Under the restricted token, Low integrity is not a total write-denial: the child can still write to the small residual Low-integrity-writable surface (`%LOCALAPPDATA%\Low`, `%TEMP%\Low`, any path with an explicit Low-integrity write ACE) and to files it creates itself.

Low integrity does not restrict reads at all, and neither does the `workspace` token below, so under the restricted token [`[permissions].confine_reads`](../usage/permissions.md#the-read-boundary) holds for the file tools alone, and startup warns once when it is on. Integrity levels stop a Low-integrity process writing up, not reading up. A command at `read` can therefore read meka's config directory and store, account credentials included, which the Bubblewrap, Landlock, macOS and AppContainer backends hide; `sandbox_backend = "appcontainer"` is what changes it. Under the restricted token, treat every credential in the store as readable by any sandboxed command, and keep meka's directories outside every workspace root, because a root's write grant covers everything under it.

#### Windows at `workspace` under the restricted token

`workspace` under the restricted token uses a second mechanism, not the Low-integrity token above; under AppContainer it is a read-write grant in the environment and nothing below applies. meka derives a capability
SID from each workspace root, places an inheritable `GENERIC_WRITE | DELETE` ACE for it on that
root, and runs the shell under a `WRITE_RESTRICTED` token carrying that capability, so a write
succeeds exactly where one of those ACEs exists. Three consequences worth knowing before you use it:

- meka has to **own** the root, which is what lets it grant without elevation. A network share or
  another user's folder cannot be a workspace root.
- PowerShell runs in **ConstrainedLanguage** mode under a restricted token, so scripts that
  construct .NET types fail there while working at `unrestricted`. meka's UTF-8 output preamble is
  skipped for the same reason, so non-ASCII output may be mangled at `workspace`.
- The ACE is real, standing state on your directory, visible in `icacls`. It is released when the
  process exits, Ctrl+C included, but not after a crash or a kill.

See [Permissions](../usage/permissions.md#per-platform-enforcement) for the full account.

#### When the configured backend is unavailable

If the configured backend can't be used (`bwrap` isn't on `$PATH`, user namespaces are denied, the Windows build lacks the AppContainer API, or the value belongs to another platform), `shell_execute` at `read` returns a hard error rather than silently falling back. The error names the configured backend, the specific failure reason and the key to fix. Either install what is missing, set `sandbox_backend` to a value this platform can use, or switch to `unrestricted` (Shift+Tab).

#### Disabling the sandbox entirely

To disable sandboxed shell execution altogether, set `sandbox = false` under `[shell]`. When disabled, shell commands require `unrestricted`: `read` loses the tool entirely, and `workspace` refuses it with an error naming the key, because there is no longer anything to hold the boundary that level promises. Reach for `unrestricted` on those turns rather than expecting `workspace` to quietly run unconfined.

```toml
[shell]
sandbox = false
```
