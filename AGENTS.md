# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Rust Coding Guidelines

- Prioritize code correctness and clarity. Speed and efficiency are secondary priorities unless otherwise specified.
- Do not write organizational or comments that summarize the code. Comments should only be written in order to explain "why" the code is written in some way in the case there is a reason that is tricky / non-obvious.
- Prefer implementing functionality in existing files unless it is a new logical component. Avoid creating many small files.
- Avoid using functions that panic like `unwrap()`, instead use mechanisms like `?` to propagate errors.
- Be careful with operations like indexing which may panic if the indexes are out of bounds.
- Never silently discard errors with `let _ =` on fallible operations. Always handle errors appropriately:
    - Propagate errors with `?` when the calling function should handle them
    - Use `.log_err()` or similar when you need to ignore errors but want visibility
    - Use explicit error handling with `match` or `if let Err(...)` when you need custom logic
    - Example: avoid `let _ = client.request(...).await?;` - use `client.request(...).await?;` instead
- When implementing async operations that may fail, ensure errors propagate to the UI layer so users get meaningful feedback.
- Never create files with `mod.rs` paths - prefer `src/some_module.rs` instead of `src/some_module/mod.rs`.
- When creating new crates, prefer specifying the library root path in `Cargo.toml` using `[lib] path = "...rs"` instead of the default `lib.rs`, to maintain consistent and descriptive naming (e.g., `gpui.rs` or `main.rs`).
- Avoid creative additions unless explicitly requested
- Use full words for variable names (no abbreviations like "q" for "queue")
- Use variable shadowing to scope clones in async contexts for clarity, minimizing the lifetime of borrowed references.
  Example:
    ```rust
    executor.spawn({
        let task_ran = task_ran.clone();
        async move {
            *task_ran.borrow_mut() = true;
        }
    });
    ```

## Logging and output

`agsh` maintains a strict split between *CLI output* and *tracing logs*. The test is simple: **if the user doesn't have to see this to use the command, it belongs in `tracing`**. Default log level is `warn`, so `info!` / `debug!` are silent unless the user passes `-v`, `-vv`, or `RUST_LOG`. Aim for "quiet on success" — the Unix convention.

**Use `println!` / `eprintln!` only when the output is unavoidable:**

- **Requested data** — what the user literally ran the command to get: the `agsh mcp list` table, `agsh mcp get` details, `agsh list` session rows, `agsh export` markdown on stdout, `print_help`.
- **Actionable content the user must copy/type/visit** — OAuth authorisation URLs, callback paste prompts, elicitation form fields, setup-wizard prompts.
- **REPL command output** — `/permission`, `/session`, `/cd` errors, `!cmd` status, tool-use indicators, streaming assistant markdown, thinking blocks, `Unknown command` feedback.
- **Hard errors** propagated back to the user with context (`render::render_error`, clap-side validation errors).
- Use `stdout` (`println!`) for parseable command output a script might consume; `stderr` (`eprintln!`) for prompts, live UI, and contract errors.

### `stdout` vs `stderr`

When `println!` / `eprintln!` *is* the right call (the output is unavoidable per the list above), the choice of stream is not a style decision — it's a contract:

- **`stdout` (`println!`, `print!`)** — only the data the user invoked the command to obtain. Examples: the agent's streamed assistant response, an `agsh list` table, an `agsh export -` markdown body, an `agsh skill show` body, `agsh mcp list` / `mcp get` / `mcp tools` rows.
- **`stderr` (`eprintln!`, `eprint!`)** — everything else: tool-call indicators, thinking blocks, todo lists, spacing newlines, status confirmations, hints, errors, interrupt notices, setup-wizard prompts, OAuth URLs, REPL UI feedback (`/permission`, `/cd`, `Unknown command`, approval prompts, `!cmd` exit-code messages).

**Litmus test:** `agsh ... 2>/dev/null | next-tool` should leave only the requested data on stdout. If a user can't usefully pipe the output, your `println!()` is probably an `eprintln!()`.

The streaming markdown renderer (`render::StreamingRenderer`) writes to stdout because the assistant response *is* the requested output for an agent turn. Every other helper in `render.rs` (`render_session_id`, `render_hint`, `render_error`, `render_thinking_block`, `render_todo_list`, `render_tool_indicator`) and every spacing-blank-line emitted around them goes to stderr.

**Use `tracing` for everything else:**

- `error!` — unrecoverable failure about to propagate up as an `AgshError`. Rare; the `?` operator usually already carries the info.
- `warn!` — recoverable fallback the user should know about by default: "failed to revoke token, continuing", "authorisation failed — rolling back", "probe: couldn't reach X". Also the right level for rollback and cleanup messages.
- `info!` — lifecycle signposts users *can* see with `-v`: "added X to config.toml", "authorized X", "connected to MCP server Y", "resuming session UUID", "auto-compacting", "exported session to path", `probe:` hints. This is the "quiet success" level — no output at default verbosity.
- `debug!` — diagnostics for module-level troubleshooting: "browser launch failed" (expected on headless), "reconnect attempt 2", raw callback parse details, `resource_metadata` URLs.

**Specifically, these informational CLI signposts are logs, not prints:**

- `ok:` confirmations (`added`, `removed`, `connected`, `authorized`, `cleared credentials`, `configuration saved`). Exit code carries success; don't reprint the command the user just ran.
- Probe results, running-OAuth banners, auto-compact hints, "resuming session: UUID", "exported to path".
- Rollback explanations ("interrupted — rolling back X", "authorisation failed — rolling back") — these are `warn!`, not print, because they are recoverable diagnostic information.

**Never mix the two:**

- Don't `eprintln!` "failed to open browser" on a fallback path when the URL is already printed — users can copy it; the warning is noise. Use `tracing::debug!`.
- Don't `tracing::info!` a command's primary output — users would need `-v` to see what they asked for.
- Don't `tracing::warn!` something that isn't a warning. Lifecycle signposts are `info!`.

**Drop redundant preambles.** If you're about to print a progress line immediately followed by the actionable info, cut the preamble. "Opening browser..." then the URL is noise; just print the URL.

**Opt-in visibility.** When a config flag like `show_session_id_on_create` explicitly requests visible output, honour it via `println!` / `eprintln!` — don't silently demote it to `info!` and force `-v`.

## CLI help text

Clap `///` doc-comments must render within 80 columns when shown via `-h`. Verify by running the actual binary for every changed subcommand — source-line length doesn't account for clap's indent, value-name length, or auto-appended hints like `[possible values: ...]`. Put `Examples:` and other long-form prose after a blank `///` line so they only show in `--help`, not `-h`. When that long-form prose has multiple lines or indented blocks (e.g. an `Examples:` list), add `#[command(verbatim_doc_comment)]` to the struct/variant so clap preserves the line breaks instead of re-wrapping them into one paragraph.

## Build & Formatting Commands

- Always run `cargo +nightly fmt` and `cargo sort -w` after editing code.
- Always run `cargo build` after completing all tasks

## Changelog

- Update `CHANGELOG.md` after every meaningful change (new features, bug fixes, breaking changes, deprecations, removals)
- Follow the [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) format
- Add entries under the `[Unreleased]` section
- Keep each changelog entry to around 100 characters

## Documentation

- Update the mdBook docs under `docs/book/src/` when adding or changing user-facing features, configuration options, CLI behavior, etc.
