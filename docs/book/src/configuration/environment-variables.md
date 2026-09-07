# Environment variables

The [config file](./config-file.md) is the recommended way to configure meka. Environment variables are useful for operational overrides; for example, in CI pipelines, containers, or to isolate a per-project config and data directory.

These operational variables override config file values but are overridden by CLI flags.

> **Accounts and profiles are not configurable via the environment.** Profile selection comes from the [config file](./config-file.md) and `--profile`; the model and every other model-tied setting come from the selected profile, the endpoint from its account; secrets come from the store via [`meka account`](./config-file.md#meka-account-cli). There are no account or profile env vars. This is deliberate: an ambient `OPENAI_API_KEY` or `MEKA_PROFILE` left in the environment must never silently rebind which account a named profile bills.

## meka-specific variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MEKA_PERMISSION` | Default permission level | `none`, `read`, `workspace`, `unrestricted` |
| `MEKA_INSTRUCTIONS` | Standing instructions as a string, overriding the `instructions.md` file. Equivalent to `--instructions`. Used by the `mekabox` container wrapper, which mounts the config directory read-only and so cannot supply a file. | `Be terse.` |
| `MEKA_INSTRUCTIONS_FILE` | Standing instructions read from this path (a file, or a directory of `*.md`). For a file you did not choose the location of, such as a Kubernetes ConfigMap. Conflicts with `MEKA_INSTRUCTIONS`. | `/run/secrets/meka-instructions` |
| `MEKA_CONFIG_DIR` | Override the default config directory. Points at the `meka` directory itself (contains `config.toml` and `skills/`). The only isolation knob that works on every platform: `dirs::config_dir()` ignores `$XDG_CONFIG_HOME` on macOS/Windows. Must be absolute; an empty or relative value is ignored with a warning rather than loading `./config.toml` from wherever meka happened to start. | `/tmp/meka-test/meka` |
| `MEKA_DATA_DIR` | Override the default data directory (where `meka.db` lives). Same cross-platform escape hatch: `dirs::data_dir()` ignores `$XDG_DATA_HOME` on macOS/Windows. Useful for tests, portable installs, and per-project session isolation. Must be absolute, for the same reason as above and more sharply: `meka.db` holds every account credential. | `/tmp/meka-test/data/meka` |
| `MEKA_SANDBOX_BACKEND` | Override `[shell].sandbox_backend` (Linux only). Pinning a value also suppresses the "install Bubblewrap" auto-resolve warning. Used by the `mekabox` wrapper to pin Landlock in the container without editing the read-only host config. | `landlock`, `bubblewrap` |
| `MEKA_RENDER_MODE` | Override `[display].render_mode`. Handy for CI / non-TTY runs that want plain output. | `syntect`, `termimad` (default), `raw` |

## MCP variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MEKA_MCP_TOOL_TIMEOUT` | Per-call timeout for MCP tools, as a duration string such as `10m` or `90s`. Applies to every remote tool invocation; on timeout meka cancels the request and returns an error to the model. A value that does not parse, or a zero, is warned about and ignored. | `10m` |

How many servers connect at once at startup is a setting, not a variable: [`[mcp].stdio_concurrency` and `[mcp].http_concurrency`](./config-file.md#mcp-top-level-table).

## Editor

| Variable | Description | Example |
|----------|-------------|---------|
| `VISUAL`, then `EDITOR` | The editor `meka memory edit` and `meka skill add --edit` open. `VISUAL` is tried first, then `EDITOR`; with neither set, `meka memory edit` fails naming both, and `meka skill add --edit` writes the skill and warns that it skipped the editor. The value is tried as a program name first and split on whitespace only if nothing is there, so `code --wait` works, and it is never run through a shell. | `nvim` |

## Logging

meka uses the `tracing` framework. The log level can be controlled with:

| Variable | Description | Example |
|----------|-------------|---------|
| `RUST_LOG` | Standard Rust log filter | `meka=debug`, `meka=trace` |

If `RUST_LOG` is not set, the verbosity flag (`-v`, `-vv`, `-vvv`) controls the level:

| Flag | Level |
|------|-------|
| (none) | `warn` |
| `-v` | `info` |
| `-vv` | `debug` |
| `-vvv` | `trace` |

Logs are written to stderr so they do not interfere with agent output.
