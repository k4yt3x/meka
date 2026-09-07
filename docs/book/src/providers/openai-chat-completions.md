# OpenAI Chat Completions

The **Chat Completions API** (`POST {base_url}/chat/completions`) with an API key. Works against OpenAI and any endpoint implementing that format: Ollama, vLLM, LM Studio, OpenRouter, Synthetic, LiteLLM.

This is *not* the legacy `/v1/completions` endpoint, which is a different protocol: a bare `prompt` string in, `choices[].text` out, no tool calling. Several of those same servers also expose it; meka does not implement it.

For the same key against OpenAI's newer protocol, see [`openai-responses`](./openai-responses.md).

## Configuration

| Setting | Value |
|---------|-------|
| Account `backend` | `openai-chat-completions` |
| Default base URL | `https://api.openai.com/v1` |
| Credential | API key (`sk-...`) kept in the store |
| Auth method | Bearer token (`Authorization: Bearer <key>`) |

### Quickest start

```bash
meka account add openai --backend openai-chat-completions
meka profile add work --account openai --model gpt-5.6-sol
```

`meka account add` prompts for your OpenAI API key, saves it to the store, and writes the
`[accounts.openai]` table. To read the key from a pipe instead of prompting, pass
`--api-key-stdin`. `meka profile add` then writes the profile that names the model.

### Config file

The two commands write this for you (the key stays in the store, not here):

```toml
default_profile = "work"

[accounts.openai]
backend = "openai-chat-completions"

[profiles.work]
account = "openai"
model   = "gpt-5.6-sol"
```

## Supported models

Any model reachable over the Chat Completions API that supports tool calling. For OpenAI's current line-up, see [OpenAI's models overview](https://platform.openai.com/docs/models); `meka profile add` suggests `gpt-5.6-sol` for a profile on an OpenAI account. Against a compatible endpoint the valid names are that server's: whatever Ollama, vLLM, LM Studio or OpenRouter serves. meka forwards the model string verbatim and doesn't gate which strings are valid.

## Custom base URL

To use an OpenAI-compatible endpoint, set the account's `base_url` when creating it:

```bash
# Ollama (no real key; pipe a placeholder)
printf 'unused' | meka account add ollama --backend openai-chat-completions \
    --base-url http://localhost:11434/v1 --api-key-stdin
meka profile add llama --account ollama --model llama3

# OpenRouter
meka account add openrouter --backend openai-chat-completions \
    --base-url https://openrouter.ai/api/v1
meka profile add sonnet --account openrouter --model anthropic/claude-sonnet-4.6
```

The resulting tables (the key, if any, lives in the store):

```toml
[accounts.ollama]
backend  = "openai-chat-completions"
base_url = "http://localhost:11434/v1"

[profiles.llama]
account = "ollama"
model   = "llama3"
```

## API details

**Endpoint:** `POST {base_url}/chat/completions`

**Tool format:** Tools are sent as function definitions:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read the contents of a file at the given path.",
    "parameters": { "type": "object", "properties": { ... } }
  }
}
```

**Tool results:** Sent back as messages with `role: "tool"` and the corresponding `tool_call_id`.

**Streaming:** Uses Server-Sent Events (SSE) with `data: {...}` lines. The stream ends with `data: [DONE]`.
