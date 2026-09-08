# Configuration overview

meka is configured with named **accounts** and **profiles** in a config file at
`~/.config/meka/config.toml`, plus secrets kept in the store. An account is where a request
goes and who meka is when it arrives: a backend, an endpoint, and the credential a login produced. A
profile is what meka asks that account for: a model and every model-tied setting. The quickest way
to get started is to let the two command suites write both for you:

```console
$ meka account add anthropic --backend claude-subscription
$ meka profile add work --account anthropic --model claude-opus-5
```

The first command writes an `[accounts.anthropic]` table to the config file, runs the OAuth login
(or prompts for an API key, depending on the backend), and saves the secret to the store. The
second writes a `[profiles.work]` table naming that account. With one profile configured, it is the
default. The resulting config looks like:

```toml
default_profile = "work"

[accounts.anthropic]
backend = "claude-subscription"

[profiles.work]
account = "anthropic"
model   = "claude-opus-5"
```

See [Config file](./config-file.md) for the full reference and the
[`meka account`](./config-file.md#meka-account-cli) and
[`meka profile`](./config-file.md#meka-profile-cli) command suites.

## Required settings

To run a turn, meka needs an active profile that names an account and a `model`, and a stored
credential for that account. If no profile can be selected, or the active profile's account has no
credential, meka prints an error pointing at `meka profile add` / `meka account login`.

| Setting | Source | Named on the command line |
|---------|--------|---------------------------|
| Profile for an existing session | The session's own row | `--profile <name>`, which **repins** the row |
| Profile for a new session | `default_profile` in config, or the sole profile | `--profile <name>` |
| Account | `[profiles.<name>].account` | none |
| Backend, endpoint, OAuth settings | `[accounts.<name>].*` | none |
| Model and every model-tied setting | `[profiles.<name>].*` | none |
| Credential (API key / OAuth) | The store, via `meka account add` / `login` | none |

## A profile is indivisible

A profile is a named bundle: the account it bills, the model, and every model-tied setting
(`context_window`, `vision`, `max_output_tokens`, `effort`, `thinking`, `thinking_budget`,
`max_request_bytes`, `thinking_display`). A session selects one by name and records that name.
**Nothing overrides a field inside one.**

There is deliberately no `--model`, `--base-url`, `--thinking` or `--thinking-budget`. A flag that
moved one field of the bundle left the rest behind, so a session could run a 200K model while
gauging its context against the 1M window its profile still stated, and never auto-compact.

To change a setting, edit the profile:

```bash
meka profile set work model claude-opus-5
```

To run something different, make a second profile on the same account and select it:

```bash
meka profile add fast --account anthropic --model claude-haiku-4-5 --context-window 200000
meka --profile fast -p "quick question"
```

## Override layers

Profile selection is layered as follows; higher-priority layers override lower ones:

1. **The session's own row**: the profile it was created with. A session that exists runs on what
   its row says, whatever `default_profile` later becomes.
2. **`--profile <name>`**: on a new session this chooses what the row records; on a resume it
   **rewrites** the row, so the change holds for every later turn and from every surface. See
   [what a resume restores](../usage/sessions.md#what-a-resume-restores).
3. **Config file**: persistent accounts and profiles in `~/.config/meka/config.toml`.
4. **Built-in defaults**: permission defaults to `read`, streaming defaults to on.

There is **no environment-variable tier** for accounts or profiles; an ambient `OPENAI_API_KEY` or
`MEKA_PROFILE` has no effect (see [Environment variables](./environment-variables.md)).

## Credential resolution

The credential for a session's profile is loaded from the store, keyed by the profile's account
name. It is acquired interactively:

- `meka account add <name>` runs the OAuth login (`claude-subscription`, `chatgpt-subscription`) or
  prompts for the API key (`anthropic-messages`, `openai-chat-completions`, `openai-responses`) when
  the account is created.
- `meka account login <name>` re-acquires it for an existing account (rotate an API key, recover
  from a dead OAuth refresh token), keeping every setting on the account and every profile on it.
  Add `--api-key-stdin` to pipe the key in for scripted rotation.
- `meka account remove <name>` deletes the stored credential and the account, once no profile names
  it.

Because secrets are keyed per account, two accounts on the same backend (for example, two Claude
subscriptions) keep independent credentials, and every profile on one account shares its login.

Deleting an `[accounts.<name>]` block by hand removes the settings but not the secret, which stays in
the store under that name. `meka account list` names any credential left that way, and `meka
account remove <name>` deletes it; see [Leftover
credentials](./config-file.md#leftover-credentials).

## Why some settings have no config key

A few things are deliberately CLI-only, with no `config.toml` key and no environment variable.
`--writable-root` is the current example: which folders a run may write at `workspace` permission is
a per-run scope, like the working directory itself, not a preference worth persisting. Writing it
into a file would make the boundary depend on where the file lives rather than on what you asked for
this time.

This is the same reasoning that keeps the working directory out of config, and it is the exception
to "config.toml is the complete source of truth": that rule covers persistent *settings*, and a
per-run scope is not one.


## When edits take effect

`meka` in the terminal reads `config.toml` and your instructions files at startup, so anything you
change applies from the next command. A long-lived host is different: `meka serve` and `meka acp`
read both **once**, when the process starts, and keep what they read for as long as they run.

Two consequences worth knowing:

- **`meka profile add` or `meka account add` while a server is running does not reach it.** The new
  entry is on disk and the listings show it, but `POST /v1/sessions` and ACP's profile picker
  answer "not configured" until the server is restarted. The same applies to editing an existing
  profile or account.
- **Editing your instructions files does not reach it either.** They are read once and go into the
  cached prompt prefix that every session shares.

Restart the server to pick either up. Everything else follows a live source and needs no restart:
skills are re-read per turn, memories per turn, and MCP tool lists follow the server.

A **rotated credential** sits between the two, and the distinction matters if you are rotating
because a key leaked. `meka account login <name>` from a second process is picked up without a
restart by anything that builds a provider *after* it: newly created sessions, ones the server
re-attaches after eviction, and ones explicitly repinned by `PATCH /v1/sessions/{id}`,
`session/set_config_option` or `/profile`.

A session already resident in memory holds the provider it was built with. For an **API-key**
account that means it keeps presenting the old key until it is evicted (`[serve] idle_timeout`, 24
hours by default) or the server restarts. For the **OAuth** backends (`claude-subscription`,
`chatgpt-subscription`) the live provider re-reads the stored bundle when it next refreshes its
token, so a rotation is usually adopted sooner, but nothing makes that happen on demand.

**To be certain a revoked credential is out of use, restart the host.**
