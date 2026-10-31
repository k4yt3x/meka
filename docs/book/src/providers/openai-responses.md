# OpenAI Responses

The **Responses API** (`POST {base_url}/responses`) with an API key. This is OpenAI's newer protocol
and the one it recommends for new work; it is also what [`chatgpt-subscription`](./chatgpt-subscription.md)
speaks, so the two differ only in how they authenticate and where they post.

## Setup

```console
$ meka account add openai --backend openai-responses
$ meka profile add work --account openai --model gpt-6-astra
```

```toml
default_profile = "work"

[accounts.openai]
backend = "openai-responses"

[profiles.work]
account = "openai"
model   = "gpt-6-astra"
```

## Configuration

### `base_url`

Defaults to `https://api.openai.com/v1`. meka appends `/responses`, so pass the base URL the server
publishes and nothing more:

```toml
base_url = "http://127.0.0.1:11434/v1"      # Ollama
base_url = "https://openrouter.ai/api/v1"   # OpenRouter
```

### `effort`

Maps to `reasoning.effort`. When unset the whole `reasoning` block is omitted and the endpoint
applies its own default. See the [`effort`](../configuration/config-file.md#effort) reference.

## Which servers serve this

| Server | Base URL | Notes |
|--------|----------|-------|
| OpenAI | `https://api.openai.com/v1` | The reference implementation |
| Ollama | `http://127.0.0.1:11434/v1` | **v0.13.3+ only**; earlier versions 404 |
| vLLM | your deployment | |
| LM Studio | your deployment | |
| OpenRouter | `https://openrouter.ai/api/v1` | Beta |
| Synthetic | not served | **Not supported**; use [`openai-chat-completions`](./openai-chat-completions.md) or [`anthropic-messages`](./anthropic-messages.md) |

Only the non-stateful flavor is needed. meka replays the whole conversation every turn and sends
`store: false`, so it never uses `previous_response_id` or server-side conversation state, which is
also all the local runtimes implement.

## Choosing between this and Chat Completions

Both take an API key and both reach most of the same servers, so the question is only which protocol
the server implements and which one you want:

- Prefer **`openai-responses`** where it is available. It is what OpenAI recommends for new work,
  and the agent tooling ecosystem has moved to it: OpenAI's own Codex CLI dropped Chat Completions
  support entirely.
- Use **`openai-chat-completions`** for a server that does not serve `/v1/responses`, which is still
  a great many of them.

Neither is the legacy `/v1/completions` endpoint, which is a third protocol with no tool calling that
meka does not implement.

## API details

**Endpoint:** `POST {base_url}/responses`
**Auth:** `Authorization: Bearer <api key>`
**Streaming:** SSE, always. `complete` folds the stream internally rather than issuing a separate
non-streaming request.

Request body fields meka sets: `model`, `input`, `instructions` (the system prompt, when non-empty),
`tools`, `tool_choice: auto`, `store: false`, `stream: true`, `reasoning.effort` (only when `effort`
is set), and `max_output_tokens` (only when `max_output_tokens` is set).

OpenAI's own fields go out too, as `chatgpt-subscription` sends them, each with a switch for an
endpoint or a model that refuses it:

| Field | What it does | Switch |
|---|---|---|
| `include: ["reasoning.encrypted_content"]` | Returns the model's reasoning sealed, which meka sends back on the turn's later requests so the model reasons on between its own tool calls | the profile's [`encrypted_reasoning`](../configuration/config-file.md#encrypted_reasoning) |
| `reasoning.summary: "auto"` | Asks for the digest meka shows as the model's thinking | the profile's [`reasoning_summary`](../configuration/config-file.md#reasoning_summary) |
| `prompt_cache_key` | Routes one conversation's requests to the cache holding its prefix; meka sends the session's id | the account's [`prompt_cache_key`](../configuration/config-file.md#prompt_cache_key) |

Most servers ignore a field they do not know, and one that refuses names the field in its error,
which is the switch to turn off. Sealed reasoning goes back only to the endpoint that sealed it, so
a session that ran on ChatGPT or another endpoint does not send that endpoint's reasoning here.
Endpoints that stream reasoning unprompted (vLLM and Ollama emit `response.reasoning_text.delta`
without being asked) render their thinking either way. The `cached_tokens` an endpoint reports
inside `input_tokens` are counted as cache reads, so `/status` shows a hit rate on this backend
too.
