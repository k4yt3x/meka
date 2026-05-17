# Introduction

**agsh** is a general-purpose AI agent runtime that provides LLMs with a rich set of tools — web search, shell execution, file editing, and more — to accomplish complex tasks. Use it as a natural-language shell, a system diagnostic helper, a research or data-analysis assistant, for general Q&A, or to add agentic capabilities to other applications.

```text
agsh [r] > find all Rust files in this project and count the lines of code
```

Instead of remembering `find . -name '*.rs' | xargs wc -l`, you describe what you want and the agent figures out how to do it.

## Features

- **Natural language interface** -- describe what you want instead of memorizing syntax
- **Built-in tools** -- file read/write/edit, glob search, regex content search (ripgrep), web fetch, web search, shell command execution
- **Scratchpad** -- session-scoped working memory for the agent to store and retrieve intermediate results
- **Sub-agents** -- delegate research tasks to sub-agents that inherit the parent's permission level
- **Multiple LLM providers** -- OpenAI API, OpenAI Codex (ChatGPT subscription), Claude API, and Claude OAuth (Claude subscription), with support for any OpenAI-compatible endpoint
- **MCP support** -- extend the agent with tools from external MCP servers
- **Permission system** -- control what the agent can do (none/read/ask/write), switchable mid-session
- **Session management** -- conversations are persisted in SQLite; resume, export, or compact any session
- **Streaming output** -- responses stream to the terminal in real time with syntax highlighting
- **Interactive and one-shot modes** -- use it as a REPL or pipe a single prompt
- **Extended thinking** -- `claude-api` and `claude-oauth` support extended thinking for complex reasoning

## How It Works

1. You type a natural language instruction
2. agsh sends it to the configured LLM along with tool definitions and a system prompt
3. The LLM decides which tools to call (if any) and returns text and/or tool calls
4. agsh executes the tool calls, feeds results back to the LLM, and repeats until the LLM is done
5. The final response is rendered as Markdown in the terminal
