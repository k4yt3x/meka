# Providers overview

A backend is how meka reaches an LLM inference service. meka ships with five, each selectable as an account's `backend`:

| Backend | Protocol | Endpoint | Auth |
|---------|----------|----------|------|
| [`anthropic-messages`](./anthropic-messages.md) | Anthropic Messages | `{base}/v1/messages` | API key |
| [`claude-subscription`](./claude-subscription.md) | Anthropic Messages | `api.anthropic.com/v1/messages` | Claude subscription |
| [`openai-chat-completions`](./openai-chat-completions.md) | OpenAI Chat Completions | `{base}/chat/completions` | API key |
| [`openai-responses`](./openai-responses.md) | OpenAI Responses | `{base}/responses` | API key |
| [`chatgpt-subscription`](./chatgpt-subscription.md) | OpenAI Responses | `chatgpt.com/backend-api/codex/responses` | ChatGPT subscription |

**A backend names the wire protocol, not a vendor.** That is deliberate, and it cuts both ways. One vendor can serve several protocols: OpenAI publishes Chat Completions *and* Responses, and they are different request shapes, not options on one. One protocol is served by many vendors: `/v1/messages` is implemented by Anthropic, Amazon Bedrock, Databricks, LiteLLM and Ollama, so calling it "the Claude API" would misname it the moment you point it elsewhere.

The two subscription backends are the exception, and carry a vendor name instead. What you pick there is a billing relationship; the endpoint and the client shape come with it and are not yours to choose.

Synthetic is the clearest case for why this matters. One vendor, two protocols, two base URLs:

```toml
[accounts.synthetic-claude]
backend  = "anthropic-messages"
base_url = "https://api.synthetic.new/anthropic/v1"

[accounts.synthetic-gpt]
backend  = "openai-chat-completions"
base_url = "https://api.synthetic.new/openai/v1"
```

## Configuring an account and a profile

A backend is reached through an **account**, which holds the endpoint and the credential, and asked
for a model through a **profile** on that account. The easiest way is the two command suites:
`meka account add` writes the account to the config file and stores the secret (API key or OAuth
token) in the store, and `meka profile add` names the model:

```console
$ meka account add anthropic --backend claude-subscription
$ meka profile add work --account anthropic --model claude-opus-5
```

This produces an `[accounts.anthropic]` and a `[profiles.work]` entry in
`~/.config/meka/config.toml`:

```toml
[accounts.anthropic]
backend = "claude-subscription"

[profiles.work]
account = "anthropic"
model   = "claude-opus-5"
```

Two profiles on one account share one login, which is how one subscription runs two models. With
one profile configured, it is the default. Once there are several, `meka profile use <name>` writes
`default_profile`; `add` never does.

## Selecting a profile

A **new** session runs on the profile named by `--profile <name>`, else `default_profile`, else the
sole profile. Switch the default with `meka profile use <name>`:

```bash
meka --profile work      # pick the profile this session starts on
meka profile use work    # persist as default_profile
```

There is no environment-variable override for profile selection.

A **resumed** session ignores all three and runs on the profile it recorded, so `meka -c` stays
where the conversation was had whatever `default_profile` currently says. `--profile` on a resume is
not a per-run override either: it **repins** the session, rewriting the row so every later resume
keeps it. `meka session list` shows which profile each session runs on, which is the whole story: a
session records a profile name and nothing else. You can move a live session with `/profile <name>`
in the REPL, `PATCH /v1/sessions/{id}` over HTTP, or the Profile picker in an ACP client. See
[Sessions](../usage/sessions.md#what-a-resume-restores).

## Pointing a backend somewhere else

Every API-key account takes a `base_url`, so the protocol you pick is independent of who serves it:

| Server | Chat Completions | Responses | Anthropic Messages |
|--------|------------------|-----------|--------------------|
| OpenAI | yes | yes | no |
| Anthropic | no | no | yes |
| Ollama | yes | yes (v0.13.3+) | yes |
| OpenRouter | yes | yes (beta) | yes |
| vLLM / LM Studio | yes | yes | no |
| Synthetic | yes | no | yes |

Where a server offers both OpenAI protocols, prefer [`openai-responses`](./openai-responses.md): it is what OpenAI recommends for new work and what the agent tooling ecosystem has moved to. Use `openai-chat-completions` for a server that does not serve Responses.

Note that several of these also expose a **legacy `/v1/completions`** endpoint. That is a third, different protocol: a bare `prompt` string in, `choices[].text` out, no tool calling. meka does not speak it. It cannot: the agent loop needs tool calls, which that protocol has no representation for.

## anthropic-messages vs claude-subscription

Both talk to Claude's `/v1/messages` endpoint, but the auth and request shape differ:

- **`anthropic-messages`** is the straightforward path: an `x-api-key` header and a plain system prompt, plus `anthropic-beta: interleaved-thinking-2025-05-14` whenever thinking is on (the default). Choose this when you have a Claude API key.
- **`claude-subscription`** replicates the Claude Code CLI exactly: OAuth tokens, fingerprint-encoded version header, xxHash64 attestation over the request body, injected billing system block. Choose this when you want to use a Claude Code subscription. Any deviation from the expected shape causes requests to be rejected, so avoid proxies that rewrite headers or reformat the body.

## Choosing between the OpenAI backends

Three backends, two protocols:

- **`openai-chat-completions`** posts to `/chat/completions` with an API key. Choose it for a server that serves only this protocol.
- **`openai-responses`** posts to `/responses` with an API key, the same protocol `chatgpt-subscription` uses. Choose it for OpenAI, or for any server that serves Responses.
- **`chatgpt-subscription`** posts to `chatgpt.com/backend-api/codex/responses`, authenticating by OAuth against `auth.openai.com` and mirroring the first-party Codex CLI. Choose it to bill a ChatGPT Plus / Pro / Team / Business subscription instead of a per-token API key.

The first two differ by protocol; the last two differ only by auth and endpoint.

## Streaming vs non-streaming

By default, meka uses streaming mode: tokens appear in the terminal as they are generated. Use `--no-stream` to wait for the complete response before displaying it.

Streaming is recommended for interactive use. Non-streaming may be useful for scripting or when the provider does not support SSE.

`--no-stream` applies to every agent in the run, sub-agents included, whichever profile a sub-agent is pinned to. Neither mode puts a clock on a reply: a stream that stays silent for five minutes is treated as dead, and a whole reply may take as long as it takes, with a connection whose peer has gone caught by TCP and HTTP/2 keepalives instead.
