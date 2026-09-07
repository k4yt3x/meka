# Config file

meka looks for a TOML configuration file at a platform-specific location:

| Platform | Path |
|----------|------|
| Linux | `~/.config/meka/config.toml` (`$XDG_CONFIG_HOME/meka/config.toml`) |
| macOS | `~/Library/Application Support/meka/config.toml` |
| Windows | `%APPDATA%\meka\config.toml` |

The config file is optional. If it does not exist, meka silently skips it.

meka refuses unknown keys: a typo (`contex_window`) or a removed key (`reasoning_effort`) fails the load with an error naming the offending key, rather than being silently ignored. Fix or remove the key to continue.

The commands that *edit* the file are exempt, so a broken config can still be repaired from the CLI: `meka mcp add` / `remove` / `enable` / `disable`, `meka account remove` and `meka profile remove` work on the raw document and don't care about an unknown key elsewhere in it. Everything that *reads* config fails instead of answering from empty defaults, because "No MCP servers." over a file full of them is indistinguishable from the truth.

Those editors only reach the keys they own, so a bad key anywhere else (`[session]`, `[permissions]`, a top-level typo, a raw syntax error) has to be fixed in an editor. The error names the file, line, column, and offending key.

Set the `MEKA_CONFIG_DIR` environment variable to override the default location entirely. The value points at the `meka` directory itself (contains `config.toml` and `skills/`). Useful for tests, portable installs, and isolating a per-project config from your global one.

Everything meka keeps in that directory is content you put there: `config.toml`, `skills/`, and `instructions.md` or `instructions/` if you use them. It is safe to keep under version control. Commands that edit the config take a cross-process lock on the directory itself, as does claiming a skill store, so neither leaves a lock file behind; a write is published by renaming a short-lived `config.toml.<pid>.<seq>.tmp` over the target, so that name can appear for the duration of one write. If you are upgrading from a version that wrote `.config.toml.lock`, or `.meka-store.lock` inside a skill store, delete them: nothing reads or writes them any more.

Windows still writes both, because its file locks are mandatory rather than advisory: a lock held on `config.toml` would make the file unreadable to the command holding it, and `LockFileEx` refuses a directory handle outright. If you keep a meka config directory under version control on Windows, ignore `.config.toml.lock` and `skills/.meka-store.lock`.

## Accounts and profiles

Where a request goes and what it asks for are configured separately. An **account** is a backend,
an endpoint and the credential a login produced: `[accounts.<name>]` in `config.toml`, with the
secret in the store under the same name. A **profile** is an account plus a model and every
model-tied setting: `[profiles.<name>]`. A session records the *profile* it runs on; the profile
names its account. Two profiles on one account share one login; one account on two endpoints is two
accounts.

**Secrets are never stored in the config file.** API keys and OAuth token bundles live in the
store, keyed by account name, and are acquired through the [`meka account`](#meka-account-cli)
command suite (`meka account add` runs the API-key prompt or the OAuth login for you). The config
file holds only the non-secret settings shown below.

```toml
default_profile = "work"

[accounts.anthropic]
backend = "claude-subscription"

[accounts.ollama]
backend  = "openai-chat-completions"
base_url = "http://localhost:11434/v1"

[profiles.work]
account = "anthropic"
model   = "claude-opus-5"

[profiles.fast]
account        = "anthropic"
model          = "claude-haiku-4-5"
context_window = 200000

[profiles.local]
account = "ollama"
model   = "llama3"
```

### Selecting the active profile

For each run meka picks one profile using this precedence:

1. `--profile <name>` CLI flag.
2. `default_profile` in the config file.
3. The sole profile, if exactly one is configured.

If none of these resolve (no profiles configured, or more than one with no `default_profile` /
`--profile`), meka errors and points you at `meka profile add` / `meka profile use`. Resuming a
session is the exception: it runs on the profile it recorded and never consults this, so an
ambiguous default does not block `meka -c`. There is no environment-variable tier for profile
selection; the config file (plus the per-run CLI flag) is the source of truth.

### Timeouts

Every backend connects with a 30-second handshake deadline and fails a stream that produces nothing
for five minutes, which surfaces as a retryable error rather than a hung turn.

Neither is a limit on the turn. There is deliberately no cap on how long a turn may run, how many
tool calls it may make, or how many tokens it may spend: those ceilings belong to your API key and
your provider plan, not to the harness. What these bound is *silence*. A model that is still
thinking is still sending, so a stream that goes quiet for five minutes has died, and waiting on it
forever is not patience.

## `default_profile`

Top-level field naming the profile to use when `--profile` isn't passed. Set it with
`meka profile use <name>`; `meka profile add` never writes it, since a sole profile is the default
by the selection rule.

## Account fields

### `backend`

The driver the account uses (required).

| Value | Protocol | Auth |
|-------|----------|------|
| `anthropic-messages` | Anthropic Messages, `POST {base}/v1/messages` | API key (`x-api-key`) |
| `claude-subscription` | Anthropic Messages, against `api.anthropic.com` | Claude subscription OAuth (fingerprinting + attestation) |
| `openai-chat-completions` | OpenAI Chat Completions, `POST {base}/chat/completions` | API key |
| `openai-responses` | OpenAI Responses, `POST {base}/responses` | API key |
| `chatgpt-subscription` | OpenAI Responses, against `chatgpt.com/backend-api/codex` | ChatGPT subscription OAuth |

An API-key backend is named for the protocol it speaks, because `base_url` decides the endpoint and
the same protocol is served by many vendors. A subscription backend is named for the product,
because the endpoint is fixed and what the account holds is a billing relationship. See [Providers
overview](../providers/overview.md) for which servers implement which protocol.

### `base_url`

Custom API base URL. Useful for:

- Self-hosted models via [Ollama](https://ollama.ai) (`http://localhost:11434/v1`)
- [OpenRouter](https://openrouter.ai) (`https://openrouter.ai/api/v1`)
- Other OpenAI-compatible API providers

If not set, defaults to:

- `https://api.openai.com/v1` for the `openai-chat-completions` and `openai-responses` backends
- `https://chatgpt.com` for the `chatgpt-subscription` backend (request path is `/backend-api/codex/responses`)
- `https://api.anthropic.com` for the `anthropic-messages` and `claude-subscription` backends

Set it with `meka account add <name> --base-url <url>`, or edit the account table by hand.

**The two API families end their base URL in different places, and that is not meka's choice.** An
OpenAI-compatible base includes the version segment, which is why every provider documents one
ending in `/v1` and why meka appends only `/chat/completions`. A Claude base is the host *root*,
because meka reaches two different roots off it: `/v1/messages` for the turn, and `/api/oauth/...`
for the subscription usage and profile endpoints. A base ending at `/v1` could not reach the second
set. The official SDKs draw the line the same way.

A gateway that fronts both APIs therefore publishes two URLs, and its Anthropic one is often written
with the `/v1` its OpenAI sibling needs (`https://api.synthetic.new/anthropic/v1`). Paste it as-is:
for an `anthropic-messages` or `claude-subscription` account meka drops a trailing `/v1`, since it
re-adds that segment on every request and the alternative is a request to `/v1/v1/messages`. Only a
trailing one goes, so a base whose path legitimately contains `/v1` earlier
(`https://gateway.ai.cloudflare.com/v1/{account}/{gateway}/anthropic`) is left alone. Trailing
slashes are trimmed for every backend.

The reverse is not inferred: an `openai-chat-completions` base is used exactly as written, because a
gateway serving `/chat/completions` at its root is legitimate and meka cannot tell that apart from a
missing `/v1`. If an OpenAI-compatible endpoint 404s, check that the base carries the version segment
its documentation shows.

### `oauth_token_url`

The OAuth token endpoint meka posts to, for the initial code exchange at `meka account add` /
`login` and for every refresh thereafter. Both, not just refreshes: it overrides a constant, so it
overrides it everywhere that constant is used. Defaults:

- `https://api.anthropic.com/v1/oauth/token` for `claude-subscription`
- `https://auth.openai.com/oauth/token` for `chatgpt-subscription`

It exists because that endpoint is the provider's fact, not meka's, and a value baked into the
binary either goes stale or sits on the far side of a proxy your network makes you use. Set it with
`client_id` when your route out needs both.

There is deliberately no `authorize_url` to go with it, and the asymmetry is not an oversight: meka
never *requests* the authorization URL, it hands it to your browser, so an egress proxy is never in
that path. The two legs meka makes itself are the code exchange and the refresh, and this covers
both.

### `client_id`

OAuth client id override (advanced; `claude-subscription` / `chatgpt-subscription` only). Leave unset to use meka's built-in default client ids.

### `device_id`

`claude-subscription` only. Stable per-device identifier embedded in `metadata.user_id` to mirror Claude Code's `~/.claude.json` device id (`getOrCreateUserID` in `utils/config.ts`).

If unset, meka first tries to adopt `userID` from `~/.claude.json` (so meka and Claude Code on the same machine look like the same device). If that file is missing or has no `userID`, meka generates a 64-character hex string. Either way, the resolved value is persisted back to the account under `[accounts.<name>].device_id`. This file write only happens for the `claude-subscription` backend; other backends don't need a device id.

You can supply your own value if you want to control attribution explicitly:

```toml
[accounts.work]
backend   = "claude-subscription"
device_id = "your-stable-id-here"
```

## Profile fields

### `account`

The account the profile bills (required). Must name an `[accounts.<name>]` table; a profile whose
account is missing is refused by name when a session tries to run on it, and `meka profile list`
says so.

### `model`

The model identifier to send to the provider, forwarded verbatim. Optional in the file, but a session cannot run without one: a profile that names no model is refused by name when a session tries to run on it. meka does not gate which strings are valid, so an OpenAI-compatible endpoint accepts whatever that server exposes.

`meka profile add` suggests `claude-opus-5` for a profile on a Claude account and `gpt-5.6-sol` for one on an OpenAI account. For the current line-ups, see [Anthropic's models overview](https://docs.claude.com/en/docs/about-claude/models/overview) and [OpenAI's models overview](https://platform.openai.com/docs/models); naming them here would go stale on someone else's schedule.

Change it with `meka profile set <name> model <value>`.

### `context_window`

The model's context window (total tokens it can hold), used for the `/status` gauge and auto-compaction. Takes precedence over [`[session].context_window`](#sessioncontext_window); when neither is set, meka assumes **1000000**.

meka never infers this from the model name and never asks the provider for it, so this is where a model smaller than the default gets stated. It is a local budgeting number that is never sent on the wire, so a wrong value can't fail a request, but leaving it at 1M for a smaller model means planned compaction never fires, and every compaction instead happens after the provider rejects the request as too large, costing a wasted round trip each time.

The window belongs to the session, not to the process: each session is measured against the profile it recorded, so two sessions in one `meka serve` can sit on profiles with different windows.

```toml
[profiles.work]
account        = "openai"
model          = "my-128k-model"
context_window = 131072
```

### `max_output_tokens`

Override the per-request output (completion) token cap. When unset, each backend keeps its built-in default:

| Backend | Default when unset |
|---|---|
| Claude, [`thinking`](#thinking) `adaptive` | 64000 |
| Claude, `budgeted` | twice the resolved budget, or 32000, whichever is larger |
| Claude, `off` | 32000 |
| Every other backend | the endpoint's own |

The Claude figures are meka's own defaults for the two Anthropic backends, taken from what Claude Code 2.1.241 sends on the wire; stating one here replaces them. The OpenAI backends send no cap unless the profile states one, because each reaches whatever `base_url` names and the endpoint's default is that endpoint's fact.

Under `thinking = "budgeted"` the value must exceed the profile's resolved thinking budget ([`thinking_budget`](#thinking_budget), else [`[thinking].budget`](#thinkingbudget), else 16000). `meka profile add` and `meka profile set` both refuse a profile that fails this, and it is validated again at startup.

```toml
[profiles.work]
account           = "anthropic"
max_output_tokens = 16000
```

### `effort`

One knob for reasoning effort across every backend: Claude sends it as `output_config.effort` (`claude-subscription` under the `effort-2025-11-24` beta, `anthropic-messages` directly), `openai-chat-completions` as `reasoning_effort` (with `max_completion_tokens` for the output cap), and the two Responses backends as `reasoning.effort` (with `max_output_tokens`).

**When unset the field is omitted, and the provider applies its own default. `claude-subscription` is the exception: it sends `high`, matching Claude Code.** That is the point of leaving it unset: effort is a request parameter the provider owns, and omitting it is how you ask for whatever that provider considers right. meka picks no tier of its own, because it cannot know which tiers a given endpoint implements: `anthropic-messages` and `openai-chat-completions` reach any compatible server, including local ones serving weights that never had a reasoning knob, and a tier the backend doesn't implement is a rejected request rather than a graceful ignore.

An explicit value is absolute: sent verbatim (trimmed and lowercased), with no validation or clamping, whatever model it is aimed at. You own correctness for your model and endpoint; an invalid value is rejected by the API. A blank value reads as unset.

Typical values: `low`, `medium`, `high`, `xhigh`, `max`.

```toml
[profiles.work]
account = "anthropic"
effort  = "xhigh"
```

### `vision`

Whether this profile's model accepts image input. Defaults to `true`. Set `false` for a text-only model so attachments are refused rather than sent to a model that cannot read them.

Refusal is per session, from the profile that session recorded, on both ACP and `POST /v1/sessions/{id}/turn`. What ACP *advertises* in `promptCapabilities.image` is necessarily per connection: `initialize` is answered before any session exists, so it reports the default profile's flag. A client on a vision-capable connection can still have its attachment refused by a session pinned to a text-only profile. See [ACP](../usage/acp.md).

```toml
[profiles.local]
account = "ollama"
model   = "llama-3-8b"
vision  = false
```

### `thinking`

Claude-only. How the request encodes extended thinking, and whether it asks for it at all:

| Value | Wire shape |
|-------|-----------|
| `adaptive` (default) | `thinking: {"type": "adaptive"}`: the model sets its own budget. Claude 4.6+ |
| `budgeted` | `thinking: {"type": "enabled", "budget_tokens": N}`, with N from [`thinking_budget`](#thinking_budget), else [`[thinking].budget`](#thinkingbudget), else 16000. Required by pre-4.6 Claude, and the form most third-party Anthropic-compatible servers implement |
| `off` | No `thinking` field |

One knob rather than two: it replaces both the old on/off switch and the encoding meka used to infer from the model name. The right value depends on the model *and* on what the endpoint implements, which meka can't determine, so the profile states it, and a profile whose `model` later changes is yours to keep correct.

```toml
[profiles.local]
account  = "gateway"
thinking = "budgeted"
```

### `thinking_budget`

Tokens the model may spend thinking. Read only under [`thinking = "budgeted"`](#thinking); the other two settings send no budget at all. A profile that states none falls back to [`[thinking].budget`](#thinkingbudget), and then to **16000**.

Per profile because it is a parameter of `thinking`, and `thinking` is per profile. It was one installation-wide value until 0.44, which meant a profile could be refused over a number stated nowhere in it, and told to fix it by lowering a global every other profile was also budgeting against. Under `thinking = "budgeted"` this profile's [`max_output_tokens`](#max_output_tokens) must exceed the resolved budget, and the remedy now names this profile's own keys.

```toml
[profiles.work]
account         = "anthropic"
thinking        = "budgeted"
thinking_budget = 20000
```

### `max_request_bytes`

Largest request body, in bytes, before the oldest tool-result images are redacted to fit; a body
that still does not fit is refused, and the turn retries without its newest attachments. Unset, the
Anthropic backends use **31457280** (30 MiB), which is Anthropic's 32 MiB cap less headroom, and the
OpenAI backends apply no ceiling until one is stated: their endpoints' caps are the endpoints' own
facts. An account reaches whatever its `base_url` names, so a profile on an endpoint with a smaller
cap states it here. Redaction removes tool-result images, oldest first; an image attached to the
newest message is never removed, and a body that still does not fit is refused so the turn can
degrade its own attachments instead. `openai-chat-completions` never sends tool-result images (that
API's tool messages are text), so there the ceiling only refuses.

```toml
[profiles.local]
account           = "gateway"
max_request_bytes = 8388608
```

### `redact_thinking`

`claude-subscription` only. Sends the `redact-thinking-2026-02-12` beta header for capable models, matching Claude Code, which enables it by default. With it on the server withholds the readable chain of thought: `thinking` blocks return with empty text plus a signature, and `redacted_thinking` blocks carry an opaque `data` payload. meka preserves and replays both verbatim, so multi-turn continuity holds. No reasoning text is shown for these models; in its place the REPL draws a live `Thinking... (150 tokens)` indicator from the server's running estimate, redrawn as the count climbs and left on screen when the phase ends, so a long silence reads as progress and stays legible afterwards. Defaults to `true`; set `false` to drop the beta and keep interleaved thinking visible.

```toml
[profiles.work]
account         = "anthropic"
redact_thinking = false
```

## `meka account` CLI

Add, re-authenticate, list and remove accounts without editing `config.toml` by hand. The
credential prompt / OAuth login runs as part of `add` and `login`, and secrets are written to the
store, never the config file.

| Command | Action |
|---|---|
| `meka account add <name> [--backend B] [--base-url U] [--client-id ID] [--oauth-token-url U] [--api-key-stdin]` | Add an account. Prompts for the backend and base URL when not flagged, then acquires the secret (OAuth login for `claude-subscription` / `chatgpt-subscription`, API-key prompt for `anthropic-messages` / `openai-chat-completions` / `openai-responses`). `--api-key-stdin` reads the key from stdin instead, and then needs `--backend` as a flag too, since a prompt would consume the piped key; it is refused for the two subscription backends, which have no key to read. `--client-id` and `--oauth-token-url` are dropped with a warning on an API-key backend, which never reads them. `device_id` has no flag, because meka resolves and persists it itself. |
| `meka account list` | List configured accounts with backend, base URL, and whether each has a stored credential; `--format json` prints the same as one document. Also names any stored credential that no account claims (see [Leftover credentials](#leftover-credentials)). |
| `meka account login <name> [--api-key-stdin]` | Re-acquire the secret for an existing account (re-authenticate, recover from a dead OAuth refresh token, or rotate an API key). `--api-key-stdin` reads the key from stdin for scripted rotation, and is refused on the subscription backends, which have no key to read. Every setting on the account is kept. |
| `meka account remove <name>` | Delete the stored credential from the store and remove the `[accounts.<name>]` entry from the config file. Refused while any profile names the account, naming the profiles: remove or repoint those first. Works on a name with only one of the two halves, so it can clean up after a hand-edit. |
| `meka account usage` / `whoami` / `stats` | The read-only account views; see [Account info](../usage/account.md). |

`--api-key-stdin` reads the key from standard input instead of prompting, for scripted setup:

```console
$ printf '%s' "$OPENAI_API_KEY" | meka account add openai --backend openai-chat-completions --api-key-stdin
```

There is no `account set`. An account has three settings a user writes, and each is the kind of
thing a login was made against, so a change is an edit to `config.toml` followed by
`meka account login <name>` when the endpoint moved.

## `meka profile` CLI

Add, switch, edit and remove profiles. A profile holds no secret, so none of these commands runs a
login.

| Command | Action |
|---|---|
| `meka profile add <name> [--account A] [--model M] [...]` | Add a profile. Prompts for the account and model when not flagged (a sole account is offered as the default; the model prompt offers `claude-opus-5` on a Claude account and `gpt-5.6-sol` on an OpenAI one), then offers an optional advanced step covering thinking, context window and effort, plus the thinking budget if you answer `budgeted`. Every other [profile field](#profile-fields) has a flag writing the key of the same name: `--context-window`, `--max-output-tokens`, `--effort`, `--vision`, `--thinking`, `--thinking-budget`, `--max-request-bytes` and `--redact-thinking`, so one non-interactive command can create a profile of any shape. An unflagged setting is left out of the profile so its documented default applies. Does not touch `default_profile`. |
| `meka profile list` | List configured profiles with account, backend, model and the default marker; `--format json` prints the same as one document. Names any profile whose account is not configured. |
| `meka profile set <name> <key> <value>` | Change one setting on an existing profile, in place. `--unset` in place of the value removes the key instead. See [Changing one setting](#changing-one-setting). |
| `meka profile use <name>` | Set `default_profile` to this profile. |
| `meka profile remove <name>` | Remove the `[profiles.<name>]` entry from the config file. Warns if it clears a `default_profile` that other profiles are still competing for, and if any sessions are pinned to the profile it deleted (those refuse to resume until it is configured again, or moved with `meka -r <id> --profile <name>`). The account and its credential stay. |

### Changing one setting

`meka profile set <name> <key> <value>` writes one key into `[profiles.<name>]`: every other
setting keeps its value, and every comment you wrote above or beside a key stays attached to that
key. This is how a profile's model changes, since there is no per-run flag for it.

Keys are left in the order [Profile fields](#profile-fields) documents, so a profile meka has
written to is in that order whatever order it was in before. That is deliberate rather than
incidental: every writer normalizes, so the file does not depend on which command last touched it,
and there is one shape to read rather than one per history. Comments move with their keys, so an
annotated profile stays annotated.

```console
$ meka profile set work model claude-opus-5
$ meka profile set work context_window 200000
$ meka profile set work effort --unset
```

`--unset` removes the key so the profile falls back to meka's default for it. That is not the same
as writing an empty value: an absent key follows whatever the documented default later becomes,
which is what an unstated setting has always meant.

Nine keys are settable, each named after the [profile field](#profile-fields) it writes:

| Key | Value |
|---|---|
| `model` | Any string, forwarded to the provider verbatim |
| `context_window` | A whole number of tokens |
| `max_output_tokens` | A whole number of tokens |
| `effort` | Any string |
| `vision` | `true` or `false` |
| `thinking` | `adaptive`, `budgeted`, or `off` |
| `thinking_budget` | A whole number of tokens |
| `max_request_bytes` | A whole number of bytes |
| `redact_thinking` | `true` or `false` |

A token count must be whole and at most 9223372036854775807, the largest integer TOML can represent;
anything else is refused before the file is opened. A boolean takes `true` or `false` and nothing
else, so `yes` and `1` are refused rather than read as true. A key that is not on the list, and a
profile name that is not configured, are both refused by name with the valid ones listed.

`account` is on the profile but deliberately not settable, and the refusal says why rather than
leaving it silently off the list: moving a profile to another account moves every session on it
onto another credential and possibly another backend. Add a profile on the other account instead.
An account key (`base_url`, `client_id`, ...) is refused with a pointer to the account table.

Two more rules are enforced on `meka profile add` and `meka profile set` alike, so neither door can
leave behind a profile the other would have declined:

- **A key on a backend that never sends it.** `thinking` and `thinking_budget` are Anthropic
  Messages request fields, so profiles on `anthropic-messages` and `claude-subscription` accounts
  carry them and nothing else does.
  [`redact_thinking`](#redact_thinking) is narrower still: it gates a beta header only
  `claude-subscription` sends, so a profile on an `anthropic-messages` account takes a thinking
  field and declines the redaction flag beside it. `set` refuses the key and writes nothing; `add`
  drops the flag with a warning and creates the profile without it. Same outcome either way: the
  key never lands where it would read plausibly and do nothing. `set --unset` is allowed on all of
  them, because removing an inert key is the remedy rather than the offense, and a hand-edited file
  is the one place one can already be sitting; such a file warns at startup. The account keys
  `client_id` and `oauth_token_url` follow the same rule on `meka account add`, which drops them
  for an API-key backend.
- **A [`max_output_tokens`](#max_output_tokens) that does not exceed the thinking budget**, under
  `thinking = "budgeted"` on one of those two backends. The budget is drawn from the output cap, so
  such a profile cannot produce a valid request; both commands check the file they are about to
  write and refuse before writing it.

### Leftover credentials

Adding an account by hand works: write an `[accounts.<name>]` block, then run `meka account login
<name>` to attach the credential. Deleting one by hand is only half the job. Credentials live in the
store keyed by account name, so removing the block takes the settings away and leaves the API
key or OAuth refresh token behind, still valid.

Nothing deletes it on your behalf. meka will not sweep the store against the config at startup:
`MEKA_CONFIG_DIR` and `MEKA_DATA_DIR` are independent, so a config read from the wrong place, or one
meka could not parse, would present as "no accounts configured" against a real store and take
every credential with it. Losing an OAuth refresh token that way means redoing the browser login for
each account.

Instead, `meka account list` reports what it finds:

```console
$ meka account list
Name  Backend             Base URL  Authenticated
work  anthropic-messages  -         yes

Stored credentials with no account: archive
```

`meka account remove archive` then deletes it. The same applies to MCP servers, reported by [`meka
mcp list`](../usage/mcp.md#meka-mcp-cli) and cleaned by `meka mcp remove <name>`.

## Examples

Each backend needs an account and then a profile on it; the account holds the login, the profile
names the model.

### `claude-subscription`

```console
$ meka account add anthropic --backend claude-subscription
# Prints the OAuth login URL for you to open, then stores the token in the store.
$ meka profile add work --account anthropic --model claude-opus-5
```

### `anthropic-messages`

```console
$ meka account add anthropic --backend anthropic-messages
# Prompts for your Anthropic API key (sk-ant-api03-...).
$ meka profile add work --account anthropic --model claude-opus-5
```

### `openai-chat-completions`

```console
$ meka account add openai --backend openai-chat-completions
# Prompts for your OpenAI API key (sk-...).
$ meka profile add work --account openai --model gpt-5.6-sol
```

### `openai-responses`

```console
$ meka account add openai --backend openai-responses
# Prompts for your OpenAI API key (sk-...). Same key as openai-chat-completions,
# newer protocol; also reaches Ollama, vLLM, LM Studio and OpenRouter.
$ meka profile add work --account openai --model gpt-5.6-sol
```

### `chatgpt-subscription`

```console
$ meka account add chatgpt --backend chatgpt-subscription
# Prints the ChatGPT OAuth login URL for you to open.
$ meka profile add work --account chatgpt --model gpt-5.6-sol
```

### Ollama (local, no key)

```console
$ printf 'unused' | meka account add ollama --backend openai-chat-completions \
    --base-url http://localhost:11434/v1 --api-key-stdin
$ meka profile add local --account ollama --model llama3
```

### OpenRouter

```console
$ meka account add openrouter --backend openai-chat-completions \
    --base-url https://openrouter.ai/api/v1
# Prompts for your OpenRouter key (sk-or-...).
$ meka profile add sonnet --account openrouter --model anthropic/claude-sonnet-4.6
$ meka profile add gpt --account openrouter --model openai/gpt-5.6-sol
```

## `[display]`

Settings for output formatting.

### `display.render_mode`

Output render mode. Equivalent to the `--render-mode` CLI flag.

| Value | Description |
|-------|-------------|
| `syntect` | Syntax-highlighted markdown source, incl. per-language code blocks; never reflowed |
| `termimad` | Rendered CommonMark, reflowed to the terminal: paragraphs re-wrap, wide tables wrap, markers are consumed. Same theme colors as `syntect`, and code blocks are highlighted by it. The default |
| `raw` | Raw markdown printed verbatim with aligned tables |

Default: `termimad`

Reflowing only happens when there is a terminal to reflow to. With output redirected or piped,
`termimad` renders without wrapping, so a captured answer is not hard-wrapped to some fallback
width.

```toml
[display]
render_mode = "raw"
```

### `display.max_width`

Widest line meka composes from model output, in terminal columns.

Default: unset, meaning the terminal's own width, so nothing ever wraps.

Set it to pin the width instead:

```toml
[display]
max_width = 120
```

A set value is honored exactly rather than clamped to the terminal, because pinning it is how you
get identical output across machines and a silent clamp would take that away on the narrow one. The
cost is that a value wider than your terminal wraps, and a wrapped row starts at column zero, where
meka's own output lives. Below 40 columns the value is clamped up and a warning is logged: every
budget subtracts fixed chrome first, and below roughly that the subtraction leaves nothing. Above
1000 it is clamped down, also with a warning, since no terminal is that wide and the value is far
more likely to be a typo than a request.

This covers meka's own output: tool indicators and their argument block, thinking previews, todo
lists, and the approval prompt. Assistant markdown is not affected and keeps reflowing to the
real terminal through [`display.render_mode`](#displayrender_mode). With output piped there is no
terminal to measure, so an unset width falls back to 100 columns and a captured run stays byte-stable.

A terminal narrower than 20 columns is treated as 20. That is not a legibility judgment: the
thinking block's own prefix is twelve columns, so below roughly that meka's chrome no longer fits and
the width stops meaning anything. Such a terminal wraps meka's output whatever the number says.

### `display.tool_params`

How much of a tool call's input the `[tool ...]` indicator shows.

This setting covers the indicator only. With approvals on, the approval prompt always shows every
argument, whatever this is set to: the indicator is a notification, the prompt is a decision,
and setting `off` for a quiet scrollback must not leave you approving calls you cannot see.

| Value | Description |
|-------|-------------|
| `off` | Name only: `[tool execute_command]`. No argument reaches your terminal |
| `summary` | Name plus the one argument that identifies the call: ``[tool execute_command(`cargo test`)]`` (default) |
| `full` | Every argument, as an indented block under the name |

Default: `summary`

`full` writes each parameter on its own line. A value that fits on a line follows its key; one that
does not gets an indented block under a bare `key:`, so a multi-line `edit_file` argument stays
readable instead of collapsing into escaped newlines. Nesting is carried by indentation, with `-`
for array elements:

```
[tool edit_file]
  path: src/render.rs
  old_string:
    let first_line = thinking.lines().next().unwrap_or("");
    let truncated = truncate_display(first_line, 80);

[tool agent_spawn]
  prompt: Audit the scheduler for missed-occurrence bugs
  tools:
    - read_file
    - search_contents
```

Consecutive calls are separated by a blank line under `full`, since each one is a block and running
them together reads as a single call with too many parameters. Under `summary` they stay flush, which
is what makes a run of them read as a list of steps.

This is a reading format, not a data format: quotes are dropped, so `timeout: 300` doesn't say
whether the model sent `300` or `"300"`. Four caps keep one call from filling the screen, and each
says what it hid:

| Cap | Limit | Marker |
|-----|-------|--------|
| One argument's value | 30 lines | `... N more lines`, indented under that argument |
| One argument's rows | 32 rows | `... N more rows`, indented under that argument |
| The block | 60 rows, checked at an argument boundary | `... N more arguments: name, name` |
| One line | [`display.max_width`](#displaymax_width) | `...` at the cut |

The first two caps look redundant and are not. A string value has lines to count, so it is trimmed
by line and the marker counts lines. An array or an object has none: it fans out one row per element,
so it needs a bound counted in rows, and the marker says rows rather than pretending they were lines.

The line cap is exact, brackets and indentation included. The block cap is not: it is checked before
an argument is rendered rather than after, so the block reaches at most the block cap plus one
argument's own budget plus the line naming what went: 93 rows.

The block cap drops whole arguments and names them rather than cutting wherever row 60 lands.
Knowing that `path` was passed but not shown beats seeing 60 rows of `content` and never learning
which file it was written to.

**A cut keeps the end.** Where a whole argument is dropped it is named; where rows are dropped the
last one is kept, so a long array still shows its final element and a trimmed value still shows how
it finishes. The reasoning is the same one that elides a long path from its middle rather than its
tail: the end of a thing too big to show is usually the half that identifies it.

When you need the exact JSON a tool was called with, `meka session export` has it, untruncated and
unflattened.

**`full` puts every argument on screen, secrets included.** `summary` shows only the one argument
that identifies a call (`write_file`'s path, `fetch_url`'s URL), so a request header carrying a token
or a file body carrying a key stayed off screen. `full` shows all of them, and replayed history
reprints them on every `/history` and every resume. meka never puts its own credentials into tool
arguments, so what appears is what the model itself passed, but that is worth knowing before turning
this on where somebody can read over your shoulder or your scrollback.

Values are escape-stripped, their newlines and carriage returns flattened, and Unicode format
characters (bidi overrides, soft hyphens, zero-width joiners) removed, so an argument cannot move
your cursor, reorder what you read, or place text at column zero where meka's own output lives.

No line exceeds [`display.max_width`](#displaymax_width), so by default nothing wraps and no row ever
begins with model text. Setting `max_width` wider than your terminal gives that up, which is the one
case where a long argument can still produce a row starting flush left.

One residual caveat: the `... N more lines`, `... N more rows` and `... N more arguments` markers are
ordinary text, so an argument whose content mimics one is indistinguishable from a real elision. That
does not let an argument run anything, but it can mislead a reader who is not expecting it.

Applies to the REPL, to one-shot runs (`meka --oneshot`), and to replayed history (`/history`,
`resume_show_recent`). ACP sends structured tool-call fields to the editor and the HTTP API's SSE
events already carry the raw input, so neither is affected.

```toml
[display]
tool_params = "full"
```

### `display.show_session_id_on_create`

Whether to display the session id when a new session is created.

Default: `false`

### `display.show_session_id_on_exit`

Whether to display the session id when meka exits.

Default: `true`

```toml
[display]
show_session_id_on_create = true
show_session_id_on_exit = false
```

### `display.show_path_in_prompt`

Whether to show the current working directory in the interactive prompt.

Default: `true`

### `display.show_context_in_prompt`

Whether to show a live context-window gauge in the interactive prompt, e.g. `128.4k/1.0M 13%` (tokens in context / model window / percent used). The figure comes from the most recent turn's reported usage (and an estimate right after `/compact` or on resume), the same value `/status` shows on its `Context:` line. Hidden until the first turn produces a measurement.

Default: `false`

### `display.newline_before_prompt`

Whether to add a blank line before the prompt, after whatever the previous line produced.

Default: `true`

### `display.newline_after_prompt`

Whether to add a blank line after the line you typed, before its output. On a resume there is no typed line: the `Continuing session:` banner takes its place, and this is the blank between that banner and the replayed history.

Default: `true`

Both apply to **anything printed between two prompts**, not only agent responses. That span is the
unit, whatever filled it: a turn, a slash command's output (`/tasks`, `/memory`, `/help`, …), an
error, a scheduled job waking the shell to run several turns at once, or any combination. It is
bracketed once, by whichever of those printed first and last, never once per turn inside it, and
never twice because two things both thought they owned the spacing.

Both space output away from *meka's* prompt, so neither applies at the edges of a run, where the
prompt is your shell's. Whatever meka prints before drawing its first prompt sits directly under the
command you typed (`Continuing session:` on a resume, or the answer to a prompt you passed on the
command line), and its last line is followed straight by the shell prompt. Start meka with no
prompt and there is nothing above its first prompt to space away from, so the rule never comes up.

The blank lines bracket output, so **a span that prints nothing gets neither**, and leaves the
screen exactly as it found it. In practice every slash command says something, even if only that a
list is empty. Three cases where nothing is printed and nothing is spaced: a successful `/cd`,
because the prompt itself is the confirmation; a successful `/clear`, because the cleared screen is;
and a scheduled wake that finds nothing left to run. `!command` is the one exception in the other
direction: it is always bracketed, because meka hands the terminal to the child process and never
learns whether it wrote anything, so a silent `!touch file` still gets its blank lines.

Turning a setting off removes that blank line and nothing else. The spacing *between* blocks of a
single response (a tool indicator and the answer that follows it, or a thinking block and the text
after it) is not controlled by either flag and does not change.

### `display.show_token_usage`

When `true`, meka prints a one-line per-turn token-usage summary to stderr after each turn:

```
[in 12.3k / cache hit 96% / out 1.2k]
```

The `in` column is the total of all three Anthropic input tiers (live, cache-write, cache-read); `cache hit %` is `cache_read / total_in`. Useful for monitoring caching effectiveness during long sessions. The `/status` slash command surfaces cumulative session stats in the same vein.

Default: `false`

### `display.stream`

Whether the answer streams to the terminal as it arrives, or lands whole when the turn ends. The
`--no-stream` flag turns streaming off for one run; this key is the standing preference.

Default: `true`

```toml
[display]
stream = false
```

### `display.resume_show_recent`

When set to a positive integer `N`, resuming a session reprints the **last `N` turns** (each turn = the user's prompt plus everything the agent did in response, styled to match the live REPL) instead of just the last assistant message.

Useful when you regularly resume long-running sessions and want more context than the single-message default. Inside a session, the `/history` slash command provides the same rendering on demand (`/history` dumps everything; `/history N` shows the last N turns).

Default: unset (resume reprints only the last assistant message, today's behavior).

```toml
[display]
resume_show_recent = 3
```

### `display.input_style`

Visual style applied to a REPL prompt once it is submitted. Makes past prompts easy to spot when scrolling back through a long session. A line still being edited keeps the terminal's own colors; the style arrives on reedline's final paint, which is the one that lands in scrollback.

The leading `/command` token is a separate signal and is colored as you type, green when meka recognizes the command and red when it does not. This setting does not affect it.

Accepted values:
- `default` (or unset): bold white-ish foreground on a slate-blue background, rendered in truecolor RGB so it looks the same across terminal themes.
- `none`: disable styling entirely.
- `reverse`: reverse video (swaps the terminal's current foreground and background).
- `bold`, `dim`, `italic`, `underline`: single attribute, no color change.
- A color name (`black`, `red`, `green`, `yellow`, `blue`, `magenta` / `purple`, `cyan`, `white`): set only the foreground, mapped to the terminal's palette.

Unknown values warn at startup and fall back to `default`.

Default: the banner preset described above.

```toml
[display]
show_path_in_prompt = false
newline_before_prompt = false
newline_after_prompt = false
input_style = "none"    # or "cyan", "bold", "dim", etc.
```

## `[web]`

Settings for the HTTP client shared by `fetch_url` and `search_web`. All keys are optional; unset fields use the defaults shown below.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `user_agent` | string | Real Chrome UA | Some search engines block non-browser UAs. Override if you need a specific identifier. |
| `request_timeout` | duration | `"30s"` | Total request budget (connect + TLS + read). `"0s"` is refused at startup. |
| `connect_timeout` | duration | unset | Separate cap on TCP + TLS handshake. Fail fast on unreachable hosts without shortening the whole request budget. `"0s"` is refused at startup. |
| `read_timeout` | duration | unset | Per-chunk idle timeout. Catches bodies that stall mid-stream. `"0s"` is refused at startup. |
| `max_redirects` | int | `10` | Cap on 3xx hops. `0` means no redirects are followed: a 3xx is returned as the response. |
| `proxy` | string | unset (honors `HTTP_PROXY` / `HTTPS_PROXY` / `ALL_PROXY` env) | Proxy URL. Schemes: `http://`, `https://`, `socks5://`, `socks5h://`, `socks4://`. The literal string `"none"` explicitly disables env-var auto-detection. |
| `ca_cert_file` | path | unset | Extra PEM bundle to trust on top of the system store. Useful for corporate MITM proxies or self-signed internal services. Accepts single-cert and multi-cert files. |
| `https_only` | bool | `false` | Refuse plain `http://` URLs. |
| `min_tls_version` | string | unset (reqwest default) | Minimum TLS version. Accepts `"1.0"`, `"1.1"`, `"1.2"`, `"1.3"`. Unknown values log a warning and fall through. Note: the bundled rustls backend supports only TLS 1.2 and 1.3; `"1.0"` / `"1.1"` will surface a build error. |
| `danger_accept_invalid_certs` | bool | `false` | **DANGEROUS.** Disable TLS certificate validation entirely. Emits a `warn!` on every startup when enabled. Only use against trusted local dev servers. |
| `danger_accept_invalid_hostnames` | bool | `false` | **DANGEROUS.** Accept certificates whose hostname doesn't match. Emits a `warn!` on every startup when enabled. Only use against trusted local dev servers. |

### Example: corporate proxy with a private CA

```toml
[web]
proxy = "http://corp-proxy.internal:3128"
ca_cert_file = "/etc/ssl/corp-root-ca.pem"
min_tls_version = "1.2"
request_timeout = "60s"
```

### Example: local testing against self-signed certs

```toml
[web]
# Route everything through a local SOCKS proxy you control.
proxy = "socks5h://127.0.0.1:1080"
# Accept self-signed certs on dev.local, KEEP THIS OFF IN PROD.
danger_accept_invalid_certs = true
```

### Example: fail-fast timeouts

```toml
[web]
request_timeout = "5s"
connect_timeout = "2s"
max_redirects = 0
```

## `[shell]`

Settings for shell command execution.

### `shell.sandbox`

Whether to enable read-only filesystem sandboxing for shell commands at `read`. When enabled (default), shell commands can be executed at `read` and `workspace` but with the filesystem write-protected outside the workspace roots. When disabled, shell commands require `unrestricted`.

Default: `true`

```toml
[shell]
sandbox = false  # disable the sandboxed shell at read
```

The sandbox uses one of two backends on Linux (see [`shell.sandbox_backend`](#shellsandbox_backend)), `sandbox-exec` on macOS, and a duplicated Low-integrity primary token on Windows. On platforms where no backend is usable, shell commands always require `unrestricted` regardless of this setting.

### `shell.sandbox_backend`

Linux-only choice between `"landlock"` and `"bubblewrap"`:

- **Bubblewrap** (`"bubblewrap"`) wraps the command in `bwrap` with read-only bind of `/`, tmpfs masks over `/run` / `/tmp` / `/var/tmp` / `$XDG_RUNTIME_DIR`, and `--unshare-user --unshare-pid --unshare-uts --unshare-ipc`. The tmpfs masks hide the dbus session bus and the systemd-user socket, so state-changing IPC calls like `systemctl --user start` and `dbus-send` fail. Network is intentionally not unshared so `curl http://x | pdftotext` still works. Requires the `bubblewrap` package and a kernel with user-namespace creation enabled.
- **Landlock** (`"landlock"`) uses the Landlock LSM to block filesystem writes, and requires **ABI v3 (kernel 6.2+)**: below that `truncate(2)` is unmediated, so a command at `read` could still empty a file, and meka reports the backend unusable instead. On kernel 7.1+ (ABI v9) it also blocks `connect()` to Unix sockets on disk, closing the dbus / systemd-user route out of the sandbox at the cost of socket-based clients like `docker` and `psql`. Between v3 and v9 that right does not exist, so a sandboxed shell can still invoke state-mutating dbus methods; meka warns at startup naming what the running ABI lacks. Kept as the lighter-weight fallback for hosts without Bubblewrap.

When omitted, meka probes Bubblewrap once at startup. If Bubblewrap is available it auto-picks it; otherwise it auto-picks Landlock and emits a one-shot warning nudging you to install `bubblewrap` for stronger protection. Set the field explicitly to either value (including `"landlock"`) to suppress that warning. No command writes this field; leave it unset to keep auto-detection.

If the configured backend can't be used at runtime (bwrap not installed, user namespaces denied, etc.), `execute_command` at `read` hard-errors with a message naming the configured backend and the specific failure reason. `read` is not blocked for other tools; only `execute_command` requires a usable sandbox.

Overridable for one run with `meka --sandbox-backend landlock|bubblewrap`, and for a whole
environment with `MEKA_SANDBOX_BACKEND`. Precedence is flag, then environment, then this field.

Default: unset (auto-detect). Ignored on macOS and Windows.

```toml
[shell]
sandbox = true
sandbox_backend = "bubblewrap"  # or "landlock"
```

## `[permissions]`

Controls which permission levels are reachable at runtime and which level the session starts at. See the [Permissions](../usage/permissions.md) page for what each level does.

| Field | Required | Description |
|-------|----------|-------------|
| `default` | No | Level the session starts at. One of `"none"`, `"read"`, `"workspace"`, `"unrestricted"`. Default `"read"`. Overridden by `--permission` and `MEKA_PERMISSION`. |
| `enabled` | No | List of levels that can be reached at runtime via `/permission` and Shift+Tab. Default `["none", "read", "workspace", "unrestricted"]`. Disabled levels are skipped during Shift+Tab cycling and refused by `/permission` with an error. |
| `approvals` | No | Whether a new session starts with [approvals](../usage/permissions.md#approvals) on: a tool call needing more than the session's level is put to you rather than refused. Default `false`. A session records its own switch afterwards, moved by `/approvals`, `PATCH /v1/sessions/{id}` or the ACP `approvals` config option. |

A level meka does not have is refused at parse, with the line. An `enabled` list that names nothing falls back to `read` alone, with a warning, rather than to the default set, so an empty list cannot widen authority. If `default` is not in `enabled`, meka logs a warning and falls back to `read` if it's enabled, otherwise the lowest enabled level (in `none → read → workspace → unrestricted` order). Same behavior if `--permission` or `MEKA_PERMISSION` selects a disabled level: meka warns and starts at the configured default rather than refusing to launch.

```toml
[permissions]
default = "read"
enabled = ["none", "read", "workspace", "unrestricted"]
approvals = true   # ask me about anything above the level instead of refusing it
```

## `[session]`

Settings for session history retention and context window management.

### `session.context_messages`

Maximum number of messages to send to the LLM API per request. Older messages are truncated from the beginning while preserving tool call chain integrity. The full history remains in the store; only the API payload is limited.

The cap is applied to every request in a turn, not just the first, so a long tool loop cannot grow the payload past it mid-turn. It is a maximum rather than a target: the cut lands on the first message that is safe to start from, which means dropping a whole `tool_use` → `tool_result` pair rather than splitting one, and a request can end up under the limit as a result. A turn whose entire tail is one unbroken tool chain is the exception; there the payload runs over rather than be rejected by the provider.

Default: `200`. `0` is refused at startup.

```toml
[session]
context_messages = 100
```

### `session.retention`

Delete sessions not updated for longer than this, at agent startup. A duration string like `"30d"` or `"12h"`. Uses `updated_at`, so an actively-resumed session is preserved even if created long ago. Deletions are reported at `warn` level.

Two kinds of session are spared whatever their timestamp says, and the sweep reports how many it left behind. A session another meka process has open is skipped: only turns bump `updated_at`, and resuming does not, so a REPL sitting at its prompt past the window looks expired while somebody is in front of it. And a session with a scheduled job still ahead of it is never expired, nor is any parent of one: a gated watcher that evaluates every tick and rarely fires looks untouched for exactly as long as it is working, and deleting it would take the schedule with it.

**Default: unset, meaning nothing is deleted.** `"0s"` is refused at startup, since it would delete everything on every launch. Conversation history isn't reproducible, so meka keeps it until told otherwise. Use `meka session delete --older-than-days <DAYS>` to prune manually instead.

```toml
[session]
retention = "30d"
```

### `session.auto_compact`

Automatically compact the conversation when input tokens exceed 80% of the context window. Compaction summarizes older messages and preserves recent ones, the todo list, and scratchpad entries.

Default: `true`

```toml
[session]
auto_compact = false
```

### `session.compact_checkpoint`

Run a *checkpoint turn* before each compaction, in which the agent saves anything that must outlive the window and writes the replacement summary itself. See [Compacting a session](../usage/sessions.md#compacting-a-session).

Costs one extra model call per compaction. Turning it off falls back to a standalone summarizer that has no tools and none of the agent's identity, so it cannot save to memory and cannot apply any judgment about what this particular agent is for.

Note that this applies to automatic compactions too, so an unattended checkpoint can write memory with nobody watching.

Default: `true`

```toml
[session]
compact_checkpoint = false
```

### `session.context_window`

Override the model's context window size (in tokens). Used for auto-compact threshold calculation. A per-profile `[profiles.<name>].context_window` takes precedence over this.

When neither is set, meka assumes **1000000**. It does not infer the window from the model name, query the provider's models API, or cache anything: the window is a local budgeting number that is never sent on the wire, so a wrong value can't fail a request, and the user is the one who knows the truth.

1M suits the current flagship models and overshoots the smaller and older ones. Overshooting is survivable rather than free: planned compaction never fires, so those sessions compact only after the provider rejects an over-long request, paying a wasted round trip each time. Set the real window on any profile whose model is smaller.

```toml
[session]
context_window = 200000
```

### `session.subagent_max_depth`

Maximum recursion depth for sub-agents spawned via [`agent_spawn`](../tools/overview.md#agent_spawn). The root agent spawns at depth 1, its sub-agents at depth 2, and so on; each level below this limit is granted its own `agent_spawn`. `1` reproduces the historical behavior where sub-agents cannot spawn further sub-agents; `0` disables `agent_spawn` entirely. An agent can tune a subtree with the tool's `max_depth` parameter, but a built-in absolute cap always bounds real nesting so recursion can't run away.

Default: `3`

```toml
[session]
subagent_max_depth = 3
```

## `[thinking]`

Presentation and budget settings for extended thinking (`anthropic-messages` and `claude-subscription` backends). Whether thinking is on, and which wire encoding it uses, is the per-profile [`thinking`](#thinking) key, not a setting here.

While the model is thinking, the REPL draws a live `Thinking...` line so a long pause reads as work rather than as a hang. On `claude-subscription` it carries the server's own running estimate (`Thinking... (150 tokens)`), redrawn in place as the count climbs; `anthropic-messages` does not report one, so the line stays bare. The count is coarse: a progress signal, not an accounting figure.

When the block ends the line stays on screen as a record that the phase happened; if the model returned readable reasoning, that text replaces the line instead. Nothing is drawn when output is piped or redirected, since there is no terminal to redraw on.

### `thinking.budget`

Maximum number of tokens the model can use for thinking. Read only under [`thinking = "budgeted"`](#thinking); the adaptive encoding lets the model set its own budget and sends no cap. A per-profile [`[profiles.<name>].thinking_budget`](#thinking_budget) takes precedence over this.

Default: `16000`

### `thinking.show_content`

Whether to show the whole text of a thinking block. When `false`, a block carrying readable reasoning is previewed as a single dimmed line, flattened across line breaks and cut to fit [`display.max_width`](#displaymax_width), and the history replayed on resume (`resume_show_recent`) omits it entirely. Emphasis on that line is styling rather than text, so a summary's `**Bold header**` reads as a bold header there too.

When `true`, the block streams to stderr as it arrives, behind the same dimmed `Thinking... ` label, with every line after the first indented by two spaces. There is no height limit: asking to see the reasoning is asking to see all of it. On a model that streams its whole chain of thought this is the difference between a token counter and the text, and the live `Thinking... (N tokens)` indicator retires as soon as the first words arrive, since the text is the better progress signal.

Formatting follows [`display.render_mode`](#displayrender_mode), with one difference: reasoning is painted entirely in dark gray, so emphasis carries as bold or italic rather than as color. That is what keeps a thinking block readable as a footnote rather than as the reply. Under `termimad` the markdown is rendered, so a reasoning summary's `**Bold header**` arrives as a bold header instead of as asterisks; under `raw` and `syntect` the source is shown as written, which for reasoning means those two produce the same output. Fenced code keeps its fences and is not syntax-highlighted, for the same reason.

Either way the block is still sent on subsequent turns, for reasoning continuity.

One cost to know about: a turn that has streamed you reasoning will not retry a transient provider failure. meka retries only while nothing the model produced has reached you, since a second attempt would repeat it, and reasoning is the first thing a turn produces. Under the default the deltas are discarded and the one-line preview is built from the completed block, so nothing is repeatable and retries behave as they always have.

Default: `false`

```toml
[thinking]
budget = 20000
show_content = true
```

## Instructions

Standing instructions are **not** a config key. They live at a conventional path beside `config.toml`, because prose long enough to be worth writing is miserable to maintain inside a TOML string:

```
~/.config/meka/
├── config.toml
├── instructions.md      # or instructions/*.md
└── skills/
```

See [Instructions](../usage/instructions.md) for the full picture. In short: write `instructions.md`, or split a large set across `instructions/*.md`, and meka reads it at startup into the `## User Instructions` section of the system prompt. To pass the text as a string instead (containers, CI), use `MEKA_INSTRUCTIONS`, `MEKA_INSTRUCTIONS_FILE`, or `--instructions`.

## `[mcp]`

Which MCP servers to connect to, and what their tools are allowed to do. The [MCP](../usage/mcp.md)
page covers the rest: the `meka mcp` command suite, where a server's secrets live, the OAuth flows,
the connection lifecycle, and the resource and prompt tools.

### `[[mcp.servers]]`

An array of MCP server configurations. Each entry defines a server to connect to at startup.

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique name for this server. Used as namespace prefix for tools (`name__tool`). Must match `[A-Za-z0-9_-]+`, must not contain `__`, and must not be `meka`, `ide`, or start with `mcp_`. |
| `transport` | Yes | Transport type: `"stdio"` (spawn subprocess) or `"http"` (streamable HTTP). |
| `command` | Stdio only | Path or name of the executable to spawn. On Windows, `npx` / `.cmd` / `.bat` / `.ps1` are auto-wrapped in `cmd /c`. |
| `args` | No | Arguments to pass to the command. |
| `env` | No | Environment variables to set for the spawned process (stdio only). The child does **not** inherit meka's environment; see below. |
| `url` | HTTP only | URL of the MCP server endpoint. |
| `auth` | No | OAuth authentication configuration (see below). Mutually exclusive with a stored bearer token. |
| `headers` | No | Custom HTTP headers to include with every request (HTTP only). |
| `headers_helper` | No | Path to an executable whose stdout (`Name: Value\n` lines) is merged over `headers` at connect-time (HTTP only). Executed with `MEKA_MCP_SERVER_NAME` / `MEKA_MCP_SERVER_URL` in env; 15 s timeout. |
| `permission` | No | Server-wide permission override: `none`, `read`, `workspace` or `unrestricted`. Applies to every tool on this server, beating the `readOnlyHint` the server advertises and the `[mcp].default_permission` global fallback. Any other value is refused at startup, naming the line, the way an unknown key is. See *Permission resolution* below. |
| `allowed_tools` | No | Optional allow-list of raw tool names (the form the server advertises, not the `server__tool` namespaced form). When set and non-empty, only these tools are registered; all others from this server are ignored. |
| `disabled_tools` | No | Optional block-list of raw tool names. Applied **after** `allowed_tools`; tools listed here are never registered. Both lists can coexist; the net set is `allowed_tools \ disabled_tools`. |
| `eager_load_tools` | No | Raw tool names that should ship **eager-loaded** instead of deferred. Listed tools skip the `load_tool` round-trip and sit in the cacheable tools-array prefix from turn 1. Use this for tools the agent invokes constantly (search, fetch, …); leave others deferred so the tools array stays lean. |
| `tool_permissions` | No | Per-tool permission overrides keyed by raw tool name, same values as `permission`. Beats the server-level `permission` and the server's `readOnlyHint` when resolving a tool's required permission. A level meka does not have is refused at startup, naming the line. |
| `trust_read_only_hint` | No | Whether this server's `readOnlyHint: true` may classify a tool as `read`. Defaults to `true`. Set `false` for a server you have not audited: its hints become advisory for display only, so its tools fall through to the strict `unrestricted` fallback, skipping `[mcp].default_permission` (a global convenience must not re-grant what a per-server audit decision refused). A `readOnlyHint: false` is still honored either way, since it only raises the requirement. See *Permission resolution* below. |
| `disabled` | No | When `true`, the server is skipped entirely at startup: no process is spawned, no HTTP connect is attempted. Flip it back with `meka mcp enable <name>` or by editing the config. Unset means `false`. |
| `required` | No | When `true`, a turn is refused while this enabled server is not `Connected` (a `disabled` server is never started, so it never gates). Over the HTTP API that refusal is a 503 `/errors/mcp-unavailable` naming the servers. When `false`, the session runs without it and its tools are simply absent. Unset inherits `[mcp].default_required` (itself `false`), so servers are optional unless they opt in. |

### `[mcp]` top-level table

| Field | Purpose |
|-------|---------|
| `default_permission` | Fallback permission for MCP tools whose server didn't advertise `readOnlyHint` and doesn't have a `permission` override. Accepts `"none"`, `"read"`, `"workspace"`, or `"unrestricted"`; any other value is refused at startup, naming the line. If unset the hardcoded fallback is `"unrestricted"` (strict). It stays there deliberately: an MCP server runs unsandboxed, so `workspace` cannot confine it. |
| `default_required` | Default for every server's `required` flag. When `true`, all enabled servers gate the turn; when `false` (the default) only servers with `required = true` do. An unavailable optional server doesn't stop the turn; its failure is logged once when it happens, and its live state is shown by `/mcp list` in the REPL or probed with `meka mcp reconnect <name>`. |
| `grace` | Per-turn cap on how long to wait for still-`Pending` servers to connect before deciding. A duration string; default `"3s"`. `"0s"` skips the wait, for scripts that want to fail fast. |
| `connect_timeout` | Per-server timeout for connect + `initialize` + `list_tools`. A hung stdio spawn or slow HTTPS handshake can't stall the whole fleet past this bound. A duration string; default `"30s"`. `"0s"` is refused at startup. |
| `stdio_concurrency` | How many stdio servers connect at once at startup. Each is a process launch, so raising it trades startup latency for load; lower it on a machine where several heavy servers starting together is the problem. Default `3`; `0` is refused at startup. |
| `http_concurrency` | How many HTTP servers connect at once at startup. Higher than the stdio limit because a connect is a request rather than a process. Default `20`; `0` is refused at startup. |

### Permission resolution

Every MCP tool's required permission is resolved through a five-step chain; the first match wins:

1. **`server.tool_permissions[<raw-tool>]`**: explicit per-tool override.
2. **`server.permission`**: explicit server-level override. Applies to every tool on that server regardless of what the server advertises.
3. **`tool.annotations.readOnlyHint`** from the server: `true` → `Read`, `false` → `Unrestricted`. The `true` half is skipped when the server sets `trust_read_only_hint = false`, and a hint skipped that way also bypasses step 4, landing on step 5.
4. **`[mcp].default_permission`**: global fallback. Not consulted for a hint that step 3 refused.
5. **Hardcoded `Unrestricted`**: strict ultimate fallback.

User-supplied config (1, 2, 4) always beats the server's self-classification; if a server lies about a tool, you can override. But when no user config says anything, the server's hint is trusted for that specific tool so `readOnlyHint = false` destructive tools don't silently become Read-accessible just because the user opted into a lenient global default.

**Hint spoofing**: `readOnlyHint` is asserted by the server and not verified by meka, and MCP tools run in the server's own process with **no sandbox**. A server that claims `readOnlyHint = true` for a tool that in fact writes therefore gets to write your tree while meka sits at `read`: MCP tools are outside the `read` filesystem boundary that covers meka's built-ins (see [Permissions](../usage/permissions.md#mcp-tools-are-the-exception)).

Three defenses, in increasing order of bluntness:

- `tool_permissions` on the specific tools you want pinned (step 1 wins).
- `trust_read_only_hint = false` on the server, which makes its hints advisory for display only. A refused hint drops straight to the strict `unrestricted` fallback, deliberately skipping `[mcp].default_permission`: that key is a global default, and letting it answer would mean `default_permission = "read"` silently re-granting exactly what the per-server flag refused. None of that server's hinted tools is reachable at `read` without an explicit override.
- `server.permission = "unrestricted"` on the whole server (step 2 wins), or `disabled_tools` to remove the tool entirely.

The hint is trusted by default because most servers annotate honestly and requiring per-tool config for every server would make `read` impractical. `trust_read_only_hint` is the switch for a server you have not audited.

**Stale config**: entries in `allowed_tools` / `disabled_tools` / `eager_load_tools` / `tool_permissions` that don't match any advertised tool get a `warn!` line at connect time. The server still connects; you just see a heads-up so you can clean up after the server renames a tool. A name that appears in both `eager_load_tools` and `disabled_tools` also warns: the disabled filter wins, so eager-loading the disabled tool is a no-op.

**Visibility across levels**: the resolved permission doesn't hide a tool from the agent. Every registered tool is listed in the per-turn context with its required level noted inline, and a `[Permission context]` section names the current level and states in one line what it allows (it does not enumerate tools; the per-tool levels are in the catalog above it). The agent can still reason about an inaccessible tool and suggest `/permission <level>` to enable it; the permission gate is enforced at dispatch time. Keeping the tool catalog visible across levels is also what lets the Claude prompt cache survive mid-session permission toggles.

#### The stdio server's environment

A stdio server is a child process that talks to the network, and it does **not** inherit meka's
environment. It receives the same curated base a shell at `read` gets (`PATH` so it can resolve its
own binaries, `HOME`, locale, `TMPDIR`), plus whatever the server's own `env` table sets.

Configuring a server is a decision to run its code, not a decision to hand it every credential on
the machine: without this, `ANTHROPIC_API_KEY`, `AWS_*` and `GITHUB_TOKEN` all rode along into every
server you had ever added.

The base also carries the machine's network configuration (`HTTP_PROXY`, `HTTPS_PROXY`, `NO_PROXY`,
`SSL_CERT_FILE`, `SSL_CERT_DIR`, `NODE_EXTRA_CA_CERTS` and the usual siblings), because a server
that cannot see them connects to nothing behind a corporate proxy and fails every call with an
error naming none of the cause. Those say where to go and whom to trust; they grant nothing.

Three families are deliberately left out and have to be requested per server: `SSH_AUTH_SOCK`, which
is a live credential agent; `NODE_OPTIONS`, which takes `--require` and therefore arbitrary code;
and the import paths `PYTHONPATH` / `NODE_PATH` / `VIRTUAL_ENV`, which change what a program loads.
A server that genuinely needs one takes it explicitly:

```toml
[[mcp.servers]]
name = "tooling"
transport = "stdio"
command = "my-tooling-server"
env = { PYTHONPATH = "${PYTHONPATH}" }
```

A server that genuinely needs a secret asks for it by name, and `${VAR}` still reads meka's
environment at connect time:

```toml
[[mcp.servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "${GITHUB_TOKEN}" }
```

#### Examples

**Exa**: reliable web search when the built-in DuckDuckGo scraper gets CAPTCHA'd. The free tier works without an API key; paste a key into the `headers` table for the paid tier:
```bash
# Free tier, no key required
meka mcp add exa https://mcp.exa.ai/mcp
```
```bash
# Paid tier, expands from EXA_API_KEY at connect time
meka mcp add exa https://mcp.exa.ai/mcp --header "x-api-key=${EXA_API_KEY}"
```

Well-annotated server: no config needed. Every tool is classified by its own `readOnlyHint` (read tools Read, write tools Write):
```toml
[[mcp.servers]]
name = "notion"
transport = "http"
url = "https://mcp.notion.com/mcp"
```

User-declared trust on an unannotated server (all tools accessible in Read):
```toml
[[mcp.servers]]
name       = "internal"
transport  = "http"
url        = "https://mcp.internal/…"
permission = "read"
```

Overriding a mis-annotated or distrusted tool (one specific tool requires `unrestricted`):
```toml
[[mcp.servers]]
name      = "notion"
transport = "http"
url       = "https://mcp.notion.com/mcp"

[mcp.servers.tool_permissions]
"notion-do-something-scary" = "unrestricted"
```

Subset of a server's tools (only `query` registers, all others are ignored):
```toml
[[mcp.servers]]
name          = "pg"
transport     = "stdio"
command       = "npx"
args          = ["-y", "@modelcontextprotocol/server-postgres"]
allowed_tools = ["query"]
```

Block-list with a narrow exception (all fs tools are Read-accessible except the two destructive ones, which are never registered):
```toml
[[mcp.servers]]
name           = "filesystem"
transport      = "stdio"
command        = "npx"
args           = ["-y", "@modelcontextprotocol/server-filesystem"]
permission     = "read"
disabled_tools = ["delete_file", "move_file"]
```

MCP tools are registered with namespaced names in the format `servername__toolname` to prevent collisions with built-in tools or between servers.

Tool and resource descriptions returned from MCP servers are truncated at 2048 characters to keep the rendered catalog bounded.

### Environment variable substitution

Every string field listed above (command, args, env values, url, headers values, `headers_helper`) supports `${VAR}` and `${VAR:-default}` expansion from the process environment. A missing variable with no default is logged at startup and left literal in `command`, `args` and `url`; in `env` or `headers`, where a credential lives, it fails closed instead: the server is marked failed and never connected, so a literal `Bearer ${TOKEN}` is not sent to anyone. Use this to avoid committing secrets:

```toml
[[mcp.servers]]
name = "github"
transport = "http"
url = "https://mcp.github.com"
headers = { X-Api-Key = "${GITHUB_MCP_TOKEN}" }
```

`env`, `args` and `headers` may *contain* a secret, but they are not one: `env` sets a subprocess's whole environment, `args` carries connection strings, and `headers` carries `X-Tenant-Id` as readily as `X-Api-Key`. meka cannot tell which is which, so they stay in `config.toml` and `${VAR}` is how you keep a value out of it.

A bearer token and an OAuth client secret are unambiguously secrets, so they are not config at all. They live in the store and are set with `meka mcp add --auth-token-stdin` / `--client-secret-stdin`, or afterwards with `meka mcp login`. See [Credentials](../usage/mcp.md#credentials).

### `[mcp.servers.auth]`

OAuth authentication for HTTP MCP servers. Set `type` to choose the authentication method. This is mutually exclusive with a stored bearer token.

The client secret is not a field here. It is a secret, so it lives in the store: set it with `meka mcp add --client-secret-stdin` or `meka mcp login <name> --client-secret-stdin`. See [Credentials](../usage/mcp.md#credentials).

| Field | Required | Description |
|-------|----------|-------------|
| `type` | Yes | Auth method: `"client_credentials"`, `"client_credentials_jwt"`, or `"oauth"` |
| `client_id` | Varies | OAuth client id (required for client_credentials/jwt, optional for oauth with dynamic registration) |
| `scopes` | No | OAuth scopes to request |
| `resource` | No | Resource parameter ([RFC 8707](https://datatracker.ietf.org/doc/html/rfc8707)), `client_credentials` and `client_credentials_jwt` only |
| `signing_key_path` | JWT only | Path to PEM private key file |
| `signing_algorithm` | No | JWT signing algorithm: `RS256` (default), `RS384`, `RS512`, `ES256`, `ES384` |
| `redirect_port` | No | Local port for OAuth authorization code callback. When omitted, meka binds to a random ephemeral port (recommended). `oauth` only. |

### Examples

#### Stdio server

```toml
[[mcp.servers]]
name = "postgres"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
permission = "unrestricted"
```

#### HTTP server

```toml
[[mcp.servers]]
name = "web-tools"
transport = "http"
url = "http://localhost:8080/mcp"
permission = "read"
```

#### HTTP server with authentication

The bearer token is not in the file. Store it once with `meka mcp add api https://api.example.com/mcp --auth-token-stdin`, or `meka mcp login api --auth-token-stdin` for a server that already exists.

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"
permission = "unrestricted"

[mcp.servers.headers]
X-Custom-Header = "value"
```

#### Stdio server with environment variables

```toml
[[mcp.servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
permission = "read"

[mcp.servers.env]
GITHUB_TOKEN = "ghp_..."
```

#### Multiple servers

```toml
[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/projects"]
permission = "read"

[[mcp.servers]]
name = "github"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
permission = "unrestricted"
```

#### HTTP server with OAuth client credentials

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"
permission = "unrestricted"

[mcp.servers.auth]
type = "client_credentials"
client_id = "my-client-id"
scopes = ["read", "write"]
```

`client_credentials` needs a client secret, which is stored rather than written here: pass `--client-secret-stdin` to `meka mcp add`, or `meka mcp login api --client-secret-stdin` afterwards.

#### HTTP server with JWT client credentials

```toml
[[mcp.servers]]
name = "api"
transport = "http"
url = "https://api.example.com/mcp"

[mcp.servers.auth]
type = "client_credentials_jwt"
client_id = "my-client-id"
signing_key_path = "/path/to/private-key.pem"
signing_algorithm = "RS256"
scopes = ["admin"]
```

#### HTTP server with OAuth authorization code flow

On first connection, meka opens a browser for authorization and stores the token for future use.

```toml
[[mcp.servers]]
name = "github-mcp"
transport = "http"
url = "https://mcp.example.com"

[mcp.servers.auth]
type = "oauth"
client_id = "my-app-id"
scopes = ["repo", "user"]
redirect_port = 8400
```

If `client_id` is omitted, meka attempts [dynamic client registration](https://datatracker.ietf.org/doc/html/rfc7591) with the server.

## `[tools]`: built-in tool filters

The three knobs `[[mcp.servers]]` exposes for MCP tools also apply to meka's built-in tools (`read_file`, `write_file`, `execute_command`, `search_web`, etc.) via a top-level `[tools]` table. MCP per-server filtering is separate from this and keeps its own namespaces; this block only affects the built-ins.

| Key | Purpose |
|---|---|
| `allowed_tools` | Optional allow-list of built-in tool names. When set and non-empty, only these built-ins register, with one exception: the seven [MCP meta-tools](../usage/mcp.md#resources-and-prompts) register regardless, because they are how the agent reaches a configured server's resources and prompts at all. Naming one here is inert and warns at startup; use `disabled_tools` to remove one. Use `meka tools list` to see the canonical names. |
| `disabled_tools` | Block-list of built-in tool names. Applied **after** `allowed_tools`; a tool here is never registered even if it also appears in the allow-list. |
| `tool_permissions` | Per-tool required-permission override keyed by built-in name. Beats the hardcoded required level from the tool's impl. Levels: `none`, `read`, `workspace`, `unrestricted`; any other value is refused at startup, naming the line. |

Stale entries (a name that matches no built-in) emit a `warn!` at startup. meka still starts; the warning just flags a likely typo or a tool the binary renamed.

Restrict a session to read-only inspection:
```toml
[tools]
allowed_tools = ["read_file", "find_files", "search_contents", "fetch_url"]
```

Force `execute_command` to need `unrestricted`, so a session below that with approvals on prompts for every shell call:
```toml
[tools.tool_permissions]
execute_command = "unrestricted"
```

Disable web access entirely in a locked-down environment:
```toml
[tools]
disabled_tools = ["search_web", "fetch_url"]
```

Sub-agents spawned via `agent_spawn` inherit the same filter; a disabled built-in is disabled everywhere. To take something away from sub-agents *only*, use [`[subagents]`](#subagents). Run `meka tools list` to see every built-in's effective required permission, whether a `[tools.tool_permissions]` override is in effect, and whether the current config enables it.

## `[subagents]`

Capabilities a sub-agent may never hold. Where `[tools]` restricts everyone, this block restricts only sub-agents.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `disabled_servers` | list | `[]` | MCP servers a sub-agent cannot see at all |
| `disabled_tools` | list | `[]` | Individual tool names a sub-agent cannot see |

```toml
[subagents]
disabled_servers = ["mekabridge"]
disabled_tools = ["mcp__notion__create_page"]
```

**`disabled_servers` is the one that matters.** Naming a server removes everything it offers from every sub-agent: its tools, its resources, and its prompts. Reach for it when a server exists to talk to *you* or to act on your behalf. The motivating case is a server that can message the user: without this, a sub-agent three levels down can send a message the user has no way to distinguish from the one they are actually talking to.

`disabled_tools` takes names as they appear in the tool list, so built-ins (`write_file`) and namespaced MCP tools (`mcp__notion__create_page`) share one namespace. For a whole server, prefer `disabled_servers`: it covers the resource and prompt surfaces that a tool-name list cannot reach.

An entry matching nothing emits a `warn!` at startup, the same way `[tools]` does. A typo here denies nothing while reading as a restriction, which is worse than writing no config at all.

These are floors. An orchestrator can restrict a particular sub-agent further with `agent_spawn`'s `deny_servers` / `deny_tools` parameters, and each level of nesting inherits everything above it, but nothing can grant back what this block took away. There is deliberately no call-site allow-list for that reason.

### Why memory and instructions are not configured here

Two things a sub-agent might inherit are deliberately absent: the memory store and the [instructions file](../usage/instructions.md). Both are granted per call by [`agent_spawn`](../tools/overview.md#agent_spawn) and default to nothing.

The distinction is what config can actually enforce. A *capability* can be withheld: a tool the registry never registered cannot be reached, however the parent phrases the task. *Context* cannot. An agent holding the instructions has them verbatim in its own system prompt, and one with `memory_read` can read any memory, so either can be copied into a sub-agent's prompt whatever config says. A `[subagents].memory = "none"` key would look like a boundary while stopping only the sub-agent's own browsing, not the content reaching it, and a control that reads as a guarantee but isn't one is worse than none.

The other half of the argument is that the config guardrail existed for a failure mode that no longer applies. It was there because the parent might *forget*, which only matters for things that are on by default. Both of these now default to off, so forgetting produces a clean sub-agent.

## `[skills]`

Controls the skill store. See the [Skills](../usage/skills.md) guide.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Register `skill_read` / `skill_search` and render the skills index |
| `agent_managed` | bool | `false` | Additionally register `skill_write` / `skill_delete` |
| `extra_paths` | array | `[]` | Additional directories to scan, read-only |

```toml
[skills]
enabled = false
```

Setting `enabled = false` keeps every skill tool's schema out of every request and renders no skills section. Files already in `~/.config/meka/skills/` are left untouched.

`agent_managed = true` lets the agent author its own skills. It is off by default because you normally curate that store yourself; it exists for a long-running agent that dispatches sub-agents, where a skill is the only artifact that both survives the session and can be handed to a sub-agent as its task. Sub-agents never receive the authoring tools whatever this is set to. See [Letting the agent manage skills](../usage/skills.md#letting-the-agent-manage-skills).

`extra_paths` adds directories to the scan. They are strictly read-only: meka never creates them and never writes into them, so an entry that does not exist is simply skipped and leaves nothing behind. A leading `~` is expanded.

```toml
[skills]
extra_paths = ["~/.agents/skills"]
```

`~/.agents/skills` is the cross-client convention, so pointing at it makes skills installed by other Agent Skills clients visible here. It is not a default: reading a directory outside meka's own namespace is your call. meka's own store is searched first and wins a name collision. There is no automatic project-level scan, for the same reason meka does not read config or instructions from the working directory; name the path here if you want a project's skills read. See [Reading skills from other directories](../usage/skills.md#reading-skills-from-other-directories).

An entry that repeats an earlier one, or that names meka's own skills directory, is dropped with a warning: it would otherwise be scanned twice and every skill in it reported as shadowed by itself. An empty string is dropped too, since it would expand to your home directory.

## `[memory]`

Controls the agent's durable note store. See the [Memory](../usage/memory.md) guide.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Register the `memory_*` tools and render the memory index |

```toml
[memory]
enabled = false
```

Setting `enabled = false` keeps the four `memory_*` tool schemas out of every request and renders no memory section, which is worth doing for lean sessions that will never use it. Memories already stored are left untouched, and `meka memory` still reaches them.

There is deliberately no environment variable and no CLI flag here: whether an agent keeps memories is a property of the installation, not something to vary per run.

## `[schedule]`

Controls the wakeups the agent schedules for itself. See the [Scheduling](../usage/scheduling.md) guide.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | Register the `schedule_*` tools and run the scheduler |
| `poll_interval` | duration | `"10s"` | How often due jobs are checked; `"0s"` is refused at startup |
| `missed_grace` | duration | `"24h"` | How late a one-shot job may be and still fire after downtime |
| `gate_timeout` | duration | `"30s"` | Wall-clock budget for a gate probe; `"0s"` is refused at startup |
| `max_jobs` | int | `50` | Per-session ceiling, refused at `schedule_create`; `0` is refused at startup |
| `max_consecutive_fires` | int | `5` | Per-session ceiling on turns spent in one sweep |
| `claim_lease` | duration | `"1h"` | How long a host's claim on a due occurrence is good for |

```toml
[schedule]
enabled = true
poll_interval = "10s"
missed_grace = "24h"
gate_timeout = "30s"
max_jobs = 50
max_consecutive_fires = 5
claim_lease = "1h"
```

`poll_interval` is the real resolution floor: a job whose interval is shorter than the tick fires once per tick, not once per interval.

`missed_grace` applies only to one-shot jobs. Recurring jobs need no equivalent, because their occurrences are one period apart, so the most recent missed one is always less than a period old; the scheduler coalesces the rest into a single catch-up fire.

`claim_lease` is how long a crashed host's occurrence stays unavailable before another host takes it. A due job is leased rather than consumed, so the row survives until the turn is delivered and a host that dies mid-delivery costs a retry rather than the occurrence. Raise it only if a gate probe plus a turn could plausibly exceed an hour; lowering it below that risks a second host taking an occurrence the first is still running, which the session lock catches at the cost of a deferral and a re-run gate probe. A host refuses to start on a value at or under `gate_timeout`, since a lease that cannot outlast the host's own probe is never right; that check does not cover the turn after the probe, which is unbounded, so leave headroom on top of it.

`max_consecutive_fires` interleaves sessions: without it, one session's whole backlog runs to completion before another session's single due job is reached. Jobs past the budget keep their occurrence, run no gate, and are taken by the next sweep most-overdue first. It bounds a batch rather than a rate: sweeps do not overlap and the next starts as soon as the last ends, so a backlog still produces one turn per job, just in interleaved groups. `0` is refused, since it would hold every job over forever; use `enabled = false` to turn scheduling off.

Setting `enabled = false` keeps the three `schedule_*` tool schemas out of every request and leaves existing jobs on disk without firing.

As with `[skills]` and `[memory]`, there is no environment variable and no CLI flag: whether an agent may schedule its own turns is a property of the installation.

## `[background]`

Controls tool calls the agent starts and does not wait for. See the [Background tasks](../usage/background.md) guide.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Offer the `background` parameter and register the `task_*` tools |
| `max_tasks` | int | `10` | Concurrent tasks per session, refused at dispatch |

```toml
[background]
enabled = true
max_tasks = 10
```

**Alone among the capability blocks, this one is off by default.** `[schedule]`, `[skills]`, and `[memory]` add capability without changing when a turn ends; this changes the contract of the primary interaction into "you asked, it answered, and something else may interrupt you later". That is right for an unattended assistant and wrong for someone using the REPL as a command line. A scheduled job also takes an explicit act to create, whereas `background` is reachable from any tool call, so an agent will reach for it unprompted.

Setting `enabled = false` keeps the `background` property out of every tool schema and the two `task_*` tools out of every request, rather than advertising a parameter that would only ever be refused.

Outcome delivery shares [`[schedule].poll_interval`](#schedule), so that key sets how long a finished task waits before it is reported, whether or not scheduling itself is enabled.

Config-only, like the blocks above: no environment variable, no CLI flag.

## `[serve]`

Configuration for `meka serve`, the HTTP API server. See the [HTTP API](../usage/http-api.md) usage guide for a full walkthrough.

### `serve.bind`

Address and port the HTTP server listens on.

| Type | Default |
|------|---------|
| `string` | `"127.0.0.1:8080"` |

```toml
[serve]
bind = "0.0.0.0:8080"
```

> **Security:** Binding to `0.0.0.0` exposes the server on all interfaces. In production, keep `127.0.0.1` and front with a TLS-terminating reverse proxy.

### `serve.max_body_bytes`

Maximum request body size in bytes. Requests exceeding this limit are refused with `413 Payload Too Large`. `0` is refused at startup; omit the field for the default.

| Type | Default |
|------|---------|
| `integer` | `10485760` (10 MiB) |

### `serve.relay_provider_errors`

Whether a 502's payload carries the provider's own response text, as a `provider_response` member
alongside `detail`.

On by default. The upstream's error type is the actionable part of a failed turn, and "consult the
server log" is no answer to anyone driving a meka they do not operate. `meka acp` honors this key too: the same policy decides what a failed turn's `error.data` carries.

What it can expose is usually the upstream's response body, which can name the *operator's*
account with the provider and its rate-limit posture: a fact about your billing relationship rather than about
the caller or the conversation, which is why this is a switch rather than a decision meka makes for
you. Not always, though. The member carries the failing call's error message, and for some failures
that is meka's own sentence about the call rather than anything the provider sent.

**It reaches `sessions:r`, not only `sessions:w`.** Submitting a turn takes the write scope, but the
failure also rides the terminal `turn.failed` event, and `GET /v1/sessions/{id}/stream` replays that
to any reader. Turn this off where read-only tokens go to people who may watch a session but are not
entitled to the account behind it.

```toml
[serve]
relay_provider_errors = false
```

`detail` is unchanged either way, so a client reading only that sees the same sentence and a
context overflow keeps its "shorten it before retrying" remedy. Off, the member is simply absent
and the text goes to the server log alone.

Bounded to the provider's own response. A required MCP server that is down still reports only the
server names under `/errors/mcp-unavailable`, because that reason is meka's own subprocess text and
has carried a command line and its filesystem path. This key does not turn that on.

| Type | Default |
|------|---------|
| `boolean` | `true` |

### `serve.docs`

Whether to serve the Swagger UI at `/v1/docs` and the OpenAPI document at `/v1/openapi.json`.

Off by default. These are the only routes on the surface that take no bearer token *and* describe
the deployment rather than report on it: what they publish is the shape of every endpoint you
expose. That is exactly what you want while building a client against a local `meka serve`, and
exactly what you do not want reachable from anywhere else. Turn it on deliberately.

```toml
[serve]
docs = true
```

| Type | Default |
|------|---------|
| `boolean` | `false` |

### `serve.max_concurrent_turns`

Process-wide cap on in-flight turns across all sessions. When the cap is reached, new turn submissions return `429 Too Many Requests` with a `Retry-After` header. Leave it **unset** for no limit; `0` is refused at startup, because a cap of zero would 429 every turn rather than mean "unlimited".

| Type | Default |
|------|---------|
| `integer` | unbounded |

### `serve.stream_replay_events`

How many SSE events per turn to retain so a client reconnecting to `GET /v1/sessions/{id}/stream` with `Last-Event-ID` can replay what it missed.

| Type | Default |
|------|---------|
| `integer` | `256` |

Matches the live broadcast channel's capacity: retaining more than the channel can buffer would let a reconnecting client replay events a *connected* consumer would have been dropped for missing. Raising it buys a longer reconnect window at the cost of per-session memory during a turn. `0` switches replay off, so a reconnect receives only what happens from then on and is told its replay is incomplete rather than being handed a silently truncated one.

### `serve.stream_reattach_grace`

How long a streaming turn keeps running after its SSE consumer disconnects, waiting for a reconnect. Accepts duration strings.

| Type | Default |
|------|---------|
| `string` (duration) | `"30s"` |

Zero subscribers means nobody is listening, and a turn with no audience is spending provider tokens for nothing. That is the right instinct and the wrong deadline: a client whose connection just dropped and one that is never coming back are the same observation until the window expires. Set `"0s"` to cancel a turn the moment its stream drops, which spends less on abandoned work and makes re-attach useful only for turns that already finished.

### `serve.idle_timeout`

How long a session can sit idle (no turns submitted) before the GC evicts it from memory. Accepts duration strings like `"24h"`, `"30m"`, `"7d"`. `"0s"` turns idle GC off: nothing is ever evicted for being idle.

| Type | Default |
|------|---------|
| `string` (duration) | `"24h"` |

Eviction drops the in-memory runtime but **preserves the SQLite row**; a later request transparently re-attaches. See `delete_on_idle` to also remove the row.

### `serve.gc_scan_interval`

How often the background GC scanner runs. Accepts duration strings; `"0s"` is refused at startup, since the scanner would then never run.

| Type | Default |
|------|---------|
| `string` (duration) | `"5m"` |

### `serve.delete_on_idle`

When `true`, idle-evicted sessions also have their SQLite row deleted. When `false` (default), only the in-memory state is dropped and the session can be re-attached later.

| Type | Default |
|------|---------|
| `bool` | `false` |

### `serve.shutdown_drain_timeout`

Maximum time to wait for in-flight turns and tasks to finish during graceful shutdown (`SIGTERM` / `SIGINT`). After this timeout, remaining tasks are aborted and the process exits.

| Type | Default |
|------|---------|
| `string` (duration) | `"30s"` |

### `[[serve.tokens]]`

An array of bearer tokens for API authentication. At least one token is required.

| Key | Required | Description |
|-----|----------|-------------|
| `token` | Yes* | The bearer token value. Supports `${ENV_VAR}` substitution. Mutually exclusive with `token_file`. |
| `token_file` | Yes* | Path to a file containing the token (one line, trimmed). Mutually exclusive with `token`. A startup warning is logged if the file is world-readable. |
| `description` | No | Human-readable label for this token (appears in logs). |
| `scopes` | Yes | Array of scope strings. One `:r` and one `:w` per subsystem: `sessions`, `skills`, `memory`, `schedule`, `mcp`. |

\* Exactly one of `token` or `token_file` must be set.

Inline plaintext tokens log a startup warning; use `${ENV_VAR}` or `token_file` for production.

#### Examples

Development token (inline):

```toml
[[serve.tokens]]
token = "sk_dev_test123"
scopes = ["sessions:r", "sessions:w"]
```

Production token (environment variable):

```toml
[[serve.tokens]]
token = "${MEKA_BRIDGE_TOKEN}"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]
```

Production token (file-based):

```toml
[[serve.tokens]]
token_file = "/etc/meka/bridge.token"
description = "telegram bridge"
scopes = ["sessions:r", "sessions:w"]
```

Admin token with every scope:

```toml
[[serve.tokens]]
token = "${MEKA_ADMIN_TOKEN}"
description = "operator"
scopes = [
    "sessions:r", "sessions:w",
    "skills:r", "skills:w",
    "memory:r", "memory:w",
    "schedule:r", "schedule:w",
    "mcp:r", "mcp:w",
]
```

Scopes are flat: `memory:r` does not imply `memory:w`, and neither implies the other. See the [HTTP API scope table](../usage/http-api.md#scopes) for what each permits. An unrecognized scope logs a warning at startup and grants nothing, so a typo like `sessions:write` is visible rather than silently inert.

### `[[serve.webhooks]]`

Outbound endpoints meka POSTs to when something happens that no client is waiting on: a scheduled job firing, a background task finishing. Omit the block entirely and meka never makes an outbound request.

```toml
[[serve.webhooks]]
url = "https://bridge.example/meka-hook"
secret = "${MEKA_WEBHOOK_SECRET}"     # or secret_file = "/etc/meka/hook.secret"
events = ["turn.finished", "turn.failed", "task.finished", "schedule.fired"]
timeout = "10s"
max_retries = 3
```

| Key | Type | Default | Notes |
|-----|------|---------|-------|
| `url` | `string` | required | `https://` or `http://`; supports `${ENV_VAR}` |
| `secret` | `string` | none | HMAC key for `X-Meka-Signature`; supports `${ENV_VAR}` |
| `secret_file` | `path` | none | Mutually exclusive with `secret`; chmod 0600 |
| `events` | `array` | required | One or more of the four names above |
| `timeout` | `duration` | `"10s"` | Per attempt; `"0s"` is refused at startup |
| `max_retries` | `integer` | `3` | Retries after the first attempt, capped at 10 |

`events` is required and every name must be recognized. An unknown event is a startup **error**, not a warning, unlike an unknown token scope: a scope that grants nothing leaves the token working for whatever else it holds, whereas an endpoint whose only subscription is a typo is silently never called at all.

Payloads carry identifiers and metadata, never message content. Omitting `secret` sends unsigned deliveries and logs a warning. See [Webhooks](../usage/http-api.md#webhooks) for the payload shape and the signature-verification recipe.
