# Accounts and profiles

An account is a backend, an endpoint and the credential a login produced; a profile is an account
plus a model and every model-tied setting. Both are written in `config.toml`, whose
[reference](../configuration/config-file.md#accounts-and-profiles) lists every key; this page is
the commands that write them and an example per backend.

## `meka account` CLI

Add, re-authenticate, list and remove accounts without editing `config.toml` by hand. The
credential prompt / OAuth login runs as part of `add` and `login`, and secrets are written to the
store, never the config file.

| Command | Action |
|---|---|
| `meka account add <name> [--backend B] [--base-url U] [--client-id ID] [--oauth-token-url U] [--interleaved-thinking BOOL] [--prompt-cache-key BOOL] [--stream-usage BOOL] [--max-tool-name-bytes BYTES] [--api-key-stdin]` | Add an account. Prompts for the backend and base URL when not flagged, then acquires the secret (OAuth login for `claude-subscription` / `chatgpt-subscription`, API-key prompt for `anthropic-messages` / `openai-chat-completions` / `openai-responses`). `--api-key-stdin` reads the key from stdin instead, and then needs `--backend` as a flag too, since a prompt would consume the piped key; it is refused for the two subscription backends, which have no key to read. `--client-id` and `--oauth-token-url` are refused on an API-key backend, which never reads them, and each of `--interleaved-thinking`, `--prompt-cache-key` and `--stream-usage` on every backend but the one that reads it, as `--max-tool-name-bytes` is on the two subscription backends. `device_id` has no flag, because meka resolves and persists it itself. |
| `meka account list` | List configured accounts with backend, base URL, and whether each has a stored credential; `--format json` prints the same as one document. Also names any stored credential that no account claims (see [Leftover credentials](#leftover-credentials)). |
| `meka account login <name> [--api-key-stdin]` | Re-acquire the secret for an existing account (re-authenticate, recover from a dead OAuth refresh token, or rotate an API key). `--api-key-stdin` reads the key from stdin for scripted rotation, and is refused on the subscription backends, which have no key to read. Every setting on the account is kept. |
| `meka account remove <name>` | Remove the `[accounts.<name>]` entry from the config file, then delete the stored credential from the store. Refused while any profile names the account, naming the profiles: remove or repoint those first. Works on a name with only one of the two halves, so it can clean up after a hand-edit, or after a removal whose credential failed to delete. |
| `meka account rename <name> <new-name>` | Rename the account in place. The `[accounts.<name>]` table keeps its position and comments, every profile naming it follows, and its stored credential moves with it, so no login is needed. Refused when the new name is taken, a leftover credential is stored under it, or another meka is refreshing the account's token at that moment. Stop running hosts first: a session a host already runs on the account refreshes its token under the old name, and that token is dropped rather than saved. `meka serve` refuses the rename itself while one may. |
| `meka account usage` / `whoami` / `stats` | The read-only account views; see [Account info](./account.md). |

`--api-key-stdin` reads the key from standard input instead of prompting, for scripted setup:

```console
$ printf '%s' "$OPENAI_API_KEY" | meka account add openai --backend openai-chat-completions --api-key-stdin
```

There is no `account set`. An account has three settings a user writes, and each is the kind of
thing a login was made against, so a change is an edit to `config.toml` followed by
`meka account login <name>` when the endpoint moved.

A running `meka serve` offers the same verbs over HTTP, on `/v1/accounts` under the `config:w`
scope; see [Managing the installation](./http-api.md#managing-the-installation).

## `meka profile` CLI

Add, switch, edit and remove profiles. A profile holds no secret, so none of these commands runs a
login.

| Command | Action |
|---|---|
| `meka profile add <name> [--account A] [--model M] [...]` | Add a profile. Prompts for the account and model when not flagged (a sole account is offered as the default; the model prompt offers `claude-opus-5-5` on a Claude account and `gpt-6-astra` on an OpenAI one), then offers an optional advanced step covering thinking, context window and effort, plus the thinking budget if you answer `budgeted`. Every other [profile field](../configuration/config-file.md#profile-fields) has a flag writing the key of the same name: `--context-window`, `--max-output-tokens`, `--effort`, `--vision`, `--thinking`, `--thinking-budget`, `--max-request-bytes`, `--cache-control`, `--thinking-display <DISPLAY>`, `--encrypted-reasoning` and `--reasoning-summary`, so one non-interactive command can create a profile of any shape. An unflagged setting is left out of the profile so its documented default applies. Writes `default_profile` only when a second profile joins a sole one, naming the one that was alone. |
| `meka profile list` | List configured profiles with account, backend, model and the default marker; `--format json` prints the same as one document. Names any profile whose account is not configured. |
| `meka profile set <name> <key> <value>` | Change one setting on an existing profile, in place. `--unset` in place of the value removes the key instead. See [Changing one setting](#changing-one-setting). |
| `meka profile use <name>` | Set `default_profile` to this profile. |
| `meka profile remove <name>` | Remove the `[profiles.<name>]` entry from the config file. Refuses the default while two or more profiles would remain, so point `default_profile` elsewhere first with `meka profile use`; with one profile left the key is dropped, since that profile is the default by the rule. Warns if any sessions are pinned to the profile it deleted (those refuse to resume until it is configured again, or moved with `meka -r <id> --profile <name>`). The account and its credential stay. |
| `meka profile rename <name> <new-name>` | Rename the profile in place. The `[profiles.<name>]` table keeps its position and comments, `default_profile` follows when it named the profile, and every session recorded on it moves, sub-agent sessions and their pinned spawn terms included. Refused when the new name is taken, when a root session already records it, or when a sub-agent is pinned to it; a sub-agent that follows its parent runs on the parent's profile whatever its row records, and is not counted. Stop running hosts first, or have `meka serve` read the file again with `POST /v1/config/reload`: a host keeps the names it last read, and its sessions on the renamed profile are refused their next turn until it reads the new one. |

A running `meka serve` offers the same verbs over HTTP, on `/v1/profiles` under the `config:w`
scope, and serves what it wrote without a restart; see
[Managing the installation](./http-api.md#managing-the-installation).

### Changing one setting

`meka profile set <name> <key> <value>` writes one key into `[profiles.<name>]`: every other
setting keeps its value, and every comment you wrote above or beside a key stays attached to that
key. This is how a profile's model changes, since there is no per-run flag for it.

Keys are left in the order [Profile fields](../configuration/config-file.md#profile-fields) documents, so a profile meka has
written to is in that order whatever order it was in before. That is deliberate rather than
incidental: every writer normalizes, so the file does not depend on which command last touched it,
and there is one shape to read rather than one per history. Comments move with their keys, so an
annotated profile stays annotated.

```console
$ meka profile set work model claude-opus-5-5
$ meka profile set work context_window 200000
$ meka profile set work effort --unset
```

`--unset` removes the key so the profile falls back to meka's default for it. That is not the same
as writing an empty value: an absent key follows whatever the documented default later becomes,
which is what an unstated setting has always meant. `model` is the one key with no default to fall
back to, so `--unset model` and an empty `model` are both refused.

Twelve keys are settable, each named after the [profile field](../configuration/config-file.md#profile-fields) it writes, in the
order the fields are documented:

| Key | Value |
|---|---|
| `model` | Any non-empty string, forwarded to the provider verbatim; the one key `--unset` refuses |
| `fallbacks` | Profile names, comma-separated; `--unset` removes the list |
| `context_window` | A whole number of tokens |
| `max_output_tokens` | A whole number of tokens |
| `max_request_bytes` | A whole number of bytes |
| `effort` | Any string |
| `vision` | `true` or `false` |
| `deferred_tools` | `native` or `append` |
| `thinking` | `adaptive`, `budgeted`, or `off` |
| `thinking_budget` | A whole number of tokens |
| `cache_control` | `true` or `false` |
| `thinking_display` | `updates`, `summarized` or `redacted` |

A token count must be whole and at most 9223372036854775807, the largest integer TOML can represent;
anything else is refused before the file is opened. A boolean takes `true` or `false` and nothing
else, so `yes` and `1` are refused rather than read as true. A key that is not on the list, and a
profile name that is not configured, are both refused by name with the valid ones listed.

`account` is on the profile but deliberately not settable, and the refusal says why rather than
leaving it silently off the list: moving a profile to another account moves every session on it
onto another credential and possibly another backend. Add a profile on the other account instead.
An account key (`base_url`, `client_id`, ...) is refused with a pointer to the account table.

Three more rules are enforced on `meka profile add` and `meka profile set` alike, so neither door
can leave behind a profile the other would have declined:

- **A profile without a model.** A session on such a profile is refused by name at its first turn,
  so `--unset model`, `set <name> model ""` and `add --model ""` are refused before the file is
  written.

- **A key on a backend that never sends it.** `thinking` and `thinking_budget` are Anthropic
  Messages request fields, so profiles on `anthropic-messages` and `claude-subscription` accounts
  carry them and nothing else does.
  [`thinking_display`](../configuration/config-file.md#thinking_display) is narrower still: it shapes a request only
  `claude-subscription` sends, so a profile on an `anthropic-messages` account takes a thinking
  field and declines the display beside it. `set` refuses the key and `add` the flag, before any
  prompt, and neither writes anything: the key never lands where it would read plausibly and do
  nothing. `set --unset` is allowed on all of
  them, because removing an inert key is the remedy rather than the offense, and a hand-edited file
  is the one place one can already be sitting; such a file warns at startup. The account keys
  `client_id` and `oauth_token_url` follow the same rule on `meka account add`, which refuses them
  for an API-key backend.
- **A [`max_output_tokens`](../configuration/config-file.md#max_output_tokens) that does not exceed the thinking budget**, under
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
mcp list`](./mcp.md#meka-mcp-cli) and cleaned by `meka mcp remove <name>`.

## Examples

Each backend needs an account and then a profile on it; the account holds the login, the profile
names the model.

### `claude-subscription`

```console
$ meka account add anthropic --backend claude-subscription
# Prints the OAuth login URL for you to open, then stores the token in the store.
$ meka profile add work --account anthropic --model claude-opus-5-5
```

### `anthropic-messages`

```console
$ meka account add anthropic --backend anthropic-messages
# Prompts for your Anthropic API key (sk-ant-api03-...).
$ meka profile add work --account anthropic --model claude-opus-5-5
```

### `openai-chat-completions`

```console
$ meka account add openai --backend openai-chat-completions
# Prompts for your OpenAI API key (sk-...).
$ meka profile add work --account openai --model gpt-6-astra
```

### `openai-responses`

```console
$ meka account add openai --backend openai-responses
# Prompts for your OpenAI API key (sk-...). Same key as openai-chat-completions,
# newer protocol; also reaches Ollama, vLLM, LM Studio and OpenRouter.
$ meka profile add work --account openai --model gpt-6-astra
```

### `chatgpt-subscription`

```console
$ meka account add chatgpt --backend chatgpt-subscription
# Prints the ChatGPT OAuth login URL for you to open.
$ meka profile add work --account chatgpt --model gpt-6-astra
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
$ meka profile add sonnet --account openrouter --model anthropic/claude-sonnet-5.5
$ meka profile add gpt --account openrouter --model openai/gpt-6-astra
```
