# llm

One chat CLI for every OpenAI-compatible API. You name a **model**; the API key
and base URL are inferred.

```
$ llm
? Model  (Use arrow keys)
 » kimi   (moonshot/kimi-k2)
   ─────────────────────────
   Moonshot (Kimi)  ·  browse models
   DeepSeek         ·  browse models
   OpenAI           ·  browse models
   ─────────────────────────
   type a model id…

moonshot/kimi-k2 via https://api.moonshot.ai/v1
/help for commands · ctrl-d to quit

> what's the capital of Sichuan?
```

## Install

```sh
uv sync
uv run llm            # or `llm` once the venv is on PATH
```

## Keys

Every provider reads one environment variable — from the environment, so
whatever loads your `.env` (mise, direnv, your shell) is what puts them there:

```sh
MOONSHOT_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

`llm providers` shows every provider, its variable, its base URL, and whether
the key is currently visible.

## How a model name resolves

`Registry.resolve()` tries three things, in order:

1. **an alias** from your config — `kimi` → `moonshot/kimi-k2`
2. **an explicit pair** — `groq/llama-3.3-70b`
3. **a known id prefix** — `kimi-k3` → moonshot, `claude-*` → anthropic,
   `gpt-*` → openai, `deepseek-*`, `grok-*`, `glm-*`, `gemini-*`, `mistral-*`,
   `sonar-*`

Check any of it without sending a request:

```sh
$ llm resolve kimi-k3
kimi-k3 →
  provider  moonshot (Moonshot (Kimi))
  model     kimi-k3
  base_url  https://api.moonshot.ai/v1
  api_key   MOONSHOT_API_KEY (set)
```

Model ids move fast, so none are hardcoded: the picker lists what each provider
returns from `GET /models`, cached for 24h.

```sh
llm models                # every provider you have a key for
llm models moonshot       # just one
llm models --refresh      # bypass the cache
```

## Web access

Models have no network. What they *can* do is emit a structured request to
call a tool; the client runs it and feeds the output back. That loop is what
`--web` turns on:

```sh
llm chat --web
llm ask --web "what changed in the latest uv release?"
```

```
> what changed in the latest uv release?
tool web_search({"query": "uv release notes latest"})
  ↳ [1] Releases · astral-sh/uv … (2140 chars, 31 lines)
tool fetch_url({"url": "https://github.com/astral-sh/uv/releases"})
  ↳ # Release 0.11.32 … (5980 chars, 112 lines)

The latest release is …
```

Two tools are offered: `web_search` and `fetch_url`. A turn is capped at
6 tool rounds. `/web` toggles it mid-conversation.

**Search backends** follow the same one-env-var pattern as providers; the
first configured one wins, and there's a keyless fallback so it works out of
the box:

| backend | env var | notes |
| --- | --- | --- |
| brave | `BRAVE_SEARCH_API_KEY` | preferred |
| tavily | `TAVILY_API_KEY` | built for LLM use |
| exa | `EXA_API_KEY` | |
| searxng | `SEARXNG_URL` | self-hosted, no key |
| duckduckgo | — | unofficial scrape, best-effort |

```sh
llm web                   # which backend is active, and why
llm web "query"           # run a search without spending tokens
llm fetch https://…       # see a page exactly as the model would
LLM_SEARCH_BACKEND=exa    # pin one explicitly
```

`fetch_url` resolves the host and refuses anything that isn't a public
address — loopback, private ranges, and link-local (including the cloud
metadata endpoint) are blocked, and every redirect hop is re-checked. The
model picks those URLs, often from a page it just read, so a prompt injected
into that page shouldn't be able to aim the fetcher at your network. Responses
are capped at 2 MB and truncated to 6 000 characters of text.

Note that some providers also do search *their* side (`web_search_options` on
OpenAI, sonar models on Perplexity, `:online` on OpenRouter). That's a
different mechanism, spelled differently by each one — this client-side loop
works identically everywhere instead.

## Commands

```sh
llm                       # pick a model, then chat
llm chat -m kimi-k3       # skip the picker
llm chat -m gpt-5 -s "answer in one sentence" -t 0.2
llm chat --web            # with web search + page fetching
llm ask "explain b-trees" -m deepseek-chat
cat main.py | llm ask "review this" -m claude-sonnet-5
llm ask --markdown "compare X and Y"   # buffer, then render markdown
llm providers
llm init                  # write a starter config
```

In chat: `/model` (reopens the picker, keeps your history), `/system`,
`/clear`, `/history`, `/save`, `/web`, `/markdown`, `/temp`, `/providers`,
`/exit`. Enter sends; alt+enter inserts a newline.

## Config

`llm init` writes `~/.config/llm/config.toml` (override with `$LLM_CONFIG`):

```toml
default_model = "moonshot/kimi-k2"

[aliases]
kimi = "moonshot/kimi-k2"
r1 = "deepseek/deepseek-reasoner"

# any OpenAI-compatible endpoint — new, or overriding a built-in
[providers.internal]
env_var = "INTERNAL_API_KEY"
base_url = "https://llm.internal.example.com/v1"
label = "Internal"
```

Aliases appear at the top of the picker. Custom providers behave exactly like
the built-ins, including live model listing.

Built in: openai, anthropic, google, moonshot, deepseek, xai, zai, mistral,
groq, cerebras, together, fireworks, deepinfra, perplexity, openrouter, ollama,
lmstudio. The two local ones need no key.

## Library use

The CLI is a thin shell over two pieces you can use directly:

```python
from llm import ChatSession, Registry
from llm.tools import web_toolbox

registry = Registry.load()
ref, client = registry.open("kimi-k3")  # a plain openai.OpenAI client

session = ChatSession(ref=ref, client=client, system="be terse")
for delta in session.stream("hello"):
    if delta.kind == "text":
        print(delta.text, end="")
```

`ChatSession` keeps the message list, tracks token usage, runs the tool loop,
and tags every streamed fragment by kind — `text`, `reasoning`, `tool_call`,
`tool_result`, `notice` — so reasoning and tool chatter never leak into the
answer.

Tools are just a schema plus a callable, so your own are a few lines:

```python
from llm.tools import Tool, ToolBox

clock = Tool(
    name="now",
    description="The current time, ISO-8601, UTC.",
    parameters={"type": "object", "properties": {}},
    run=lambda: datetime.now(UTC).isoformat(),
)
session.tools = ToolBox([*web_toolbox().tools, clock])
```

`ToolBox.call` never raises — a failing tool returns its error as text so the
model can recover instead of the turn dying.

## Development

```sh
uv run pytest
uv run ruff check . && uv run ruff format .
uv run pyrefly check
```
