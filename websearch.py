"""
Chat with a local llama.cpp server, giving the model a web_search tool.

The model (e.g. Qwen3) decides when to search; THIS script runs the actual
search via DuckDuckGo (no API key) and feeds results back. llama.cpp only
speaks the tool-calling protocol — the search execution lives here.

Start your server first:
    ./build/bin/llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M

Then run:
    uv run websearch.py
"""

import json
import os
import sys

import httpx
import lxml.html
from ddgs import DDGS
from openai import OpenAI

# --- config -----------------------------------------------------------------
BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080/v1")
MODEL = os.environ.get("LLAMA_MODEL", "qwen3")  # name is ignored by llama-server
MAX_RESULTS = 5
MAX_TOOL_ROUNDS = 5  # safety cap so a tool loop can't run forever
FETCH_MAX_CHARS = 6000  # cap page text so one page can't blow out the context
FETCH_TIMEOUT = 15  # seconds
SHOW_THINKING = os.environ.get("SHOW_THINKING", "1") != "0"  # print model reasoning

client = OpenAI(base_url=BASE_URL, api_key="sk-no-key-needed")

# --- the tool the model is allowed to call ----------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current/up-to-date information. "
                "Use this when the answer may be newer than your training data "
                "or when asked about recent events, prices, versions, or news."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a web page and return its readable text content. "
                "Use this after web_search to read a promising result in full, "
                "since search only returns short snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch (must start with http)",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


def web_search(query: str) -> str:
    """Run an actual DuckDuckGo search and return formatted results."""
    print(f"  \033[33m[searching: {query}]\033[0m")
    try:
        hits = DDGS().text(query, max_results=MAX_RESULTS)
    except Exception as e:  # network hiccup, rate limit, etc.
        return f"Search failed: {e}"
    if not hits:
        return "No results found."
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h.get('title', '')}\n{h.get('href', '')}\n{h.get('body', '')}")
    return "\n\n".join(lines)


def fetch_url(url: str) -> str:
    """Fetch a page and return its readable text (scripts/styles stripped)."""
    print(f"  \033[33m[fetching: {url}]\033[0m")
    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"
    try:
        resp = httpx.get(
            url,
            timeout=FETCH_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (llama-websearch)"},
        )
        resp.raise_for_status()
    except Exception as e:  # timeout, DNS, HTTP error, etc.
        return f"Fetch failed: {e}"

    ctype = resp.headers.get("content-type", "")
    if "html" not in ctype and "text" not in ctype:
        return f"Unsupported content type: {ctype or 'unknown'}"

    try:
        tree = lxml.html.fromstring(resp.text)
        # drop noise that isn't readable content
        for bad in tree.xpath("//script | //style | //noscript | //nav | //footer | //header"):
            bad.getparent().remove(bad)
        text = tree.text_content()
    except Exception as e:
        return f"Parse failed: {e}"

    # collapse whitespace so the model isn't fed acres of blank lines
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text) > FETCH_MAX_CHARS:
        text = text[:FETCH_MAX_CHARS] + "\n...[truncated]"
    return text or "(no readable text found on page)"


# map tool name -> python function
TOOL_IMPLS = {"web_search": web_search, "fetch_url": fetch_url}


def run_turn(messages: list) -> str:
    """Send messages, handle any tool calls, return the final assistant text."""
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # llama-server (reasoning-format auto) returns the model's thinking in a
        # separate reasoning_content field — print it dimmed if present.
        if SHOW_THINKING:
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                print(f"\033[2m[thinking] {reasoning.strip()}\033[0m\n")

        if not msg.tool_calls:
            return msg.content or ""

        # the model asked to call one or more tools — run each and feed back
        for call in msg.tool_calls:
            fn = TOOL_IMPLS.get(call.function.name)
            if fn is None:
                result = f"Unknown tool: {call.function.name}"
            else:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = fn(**args)
                except Exception as e:
                    result = f"Tool error: {e}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                }
            )
    return "(stopped: too many tool-call rounds)"


def main() -> None:
    print("llama.cpp + web search. Type a question, or Ctrl+C / 'exit' to quit.\n")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the web_search tool for "
            "anything that needs current information, then use fetch_url to read "
            "the most promising result in full when snippets aren't enough. "
            "Cite the sources you used.",
        }
    ]
    while True:
        try:
            user = input("\033[36myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        answer = run_turn(messages)
        print(f"\n\033[32mbot>\033[0m {answer}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
