# llama-websearch

Give a local [llama.cpp](https://github.com/ggml-org/llama.cpp) model a `web_search`
tool. The model decides *when* to search; this script runs the actual search via
DuckDuckGo (no API key) and feeds the results back. llama.cpp only speaks the
OpenAI-compatible tool-calling protocol — the search execution lives in the script.

## Prerequisites

A running llama.cpp server with a tool-call-capable model:

```sh
./build/bin/llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M
```

(leave it running in another terminal — it listens on http://localhost:8080)

## Run

```sh
uv run websearch.py
```

Then chat. Ask something current ("what's the latest llama.cpp release?") and
you'll see a yellow `[searching: ...]` line when the model decides to search.

## Config (env vars)

| Var               | Default                      | Notes                              |
|-------------------|------------------------------|------------------------------------|
| `LLAMA_BASE_URL`  | `http://localhost:8080/v1`   | point at a different server/port   |
| `LLAMA_MODEL`     | `qwen3`                      | ignored by llama-server (cosmetic) |
| `SHOW_THINKING`   | `1`                          | set to `0` to hide model reasoning |

The model's reasoning (Qwen3's `<think>` content) prints dimmed as `[thinking] ...`.
It comes from llama-server's `message.reasoning_content` field (enabled by the
default `--reasoning-format auto`).

## Tools the model can call

- `web_search(query)` — DuckDuckGo search, returns titles/links/snippets.
- `fetch_url(url)` — fetches a page and returns its readable text (HTML stripped,
  capped at ~6000 chars) so the model can read a result in full, not just a snippet.

## How it works

1. Each request declares the tools in the `tools` array.
2. The model returns a tool call (search, then optionally fetch) instead of answering.
3. This script runs the tool and returns the result as a `tool` message.
4. The model reads the result and answers — looping (up to 5 rounds) if it wants
   to search again or fetch a page before responding.
