# llm

One chat CLI for every OpenAI-compatible API. You name a **model**; the API key
and base URL are inferred.

```shell
$ uv run llm
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

> what's the capital of France?
```

## Install

```shell
uv run llm
```

## Keys

Every provider reads one environment variable from the environment. They can be loaded from a `.env` file by using tools like mise, direnv or directly exporting them in your shell.

```shell
MOONSHOT_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

`llm providers` shows every provider, its variable, its base URL, and whether
the key is currently visible.

## Model name resolution

`Registry.resolve()` tries three things, in order:

1. An alias from your config, e.g. `kimi` → `moonshot/kimi-k2`
2. An explicit pair, e.g. `groq/llama-3.3-70b`
3. A known id prefix, e.g. `kimi-k3` → moonshot, `claude-*` → anthropic, `gpt-*` → openai, `deepseek-*`, `grok-*`, `glm-*`, `gemini-*`, `mistral-*`, `sonar-*`

Check any of it without sending a request:

```shell
$ uv run llm resolve kimi-k3
kimi-k3 →
  provider  moonshot (Moonshot (Kimi))
  model     kimi-k3
  base_url  https://api.moonshot.ai/v1
  api_key   MOONSHOT_API_KEY (set)
```

Model ids move fast, so none are hardcoded: the picker lists what each provider
returns from `GET /models`, cached for 24h.

```shell
uv run llm models                # every provider you have a key for
uv run llm models moonshot       # just one
uv run llm models --refresh      # bypass the cache
```

## Web access

Models have no network. What they _can_ do is emit a structured request to
call a tool; the client runs it and feeds the output back:

```shell
$ uv run llm chat --web
$ uv run llm ask --web "what changed in the latest uv release?"
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
first configured one wins, and there's a keyless fallback so it works out of the box.

| backend    | env var                | notes                          |
| ---------- | ---------------------- | ------------------------------ |
| brave      | `BRAVE_SEARCH_API_KEY` | preferred                      |
| tavily     | `TAVILY_API_KEY`       | built for LLM use              |
| exa        | `EXA_API_KEY`          |                                |
| searxng    | `SEARXNG_URL`          | self-hosted, no key            |
| duckduckgo | —                      | unofficial scrape, best-effort |

```shell
uv run llm web                   # which backend is active, and why
uv run llm web "query"           # run a search without spending tokens
uv run llm fetch https://…       # see a page exactly as the model would
LLM_SEARCH_BACKEND=exa    # pin one explicitly
```

`fetch_url` resolves the host and refuses anything that isn't a public
address. Loopback, private ranges, and link-local (including the cloud
metadata endpoint) are blocked, and every redirect hop is re-checked. The
model picks those URLs, often from a page it just read, so a prompt injected into that page shouldn't be able to aim the fetcher at your network. Responses are capped at 2 MB and truncated to 6,000 characters of text.

Note that some providers also do search _their_ side (`web_search_options` on OpenAI, sonar models on Perplexity, `:online` on OpenRouter). That's a different mechanism, spelled differently by each one. This client-side loop works identically everywhere instead.

## Commands

```shell
uv run llm providers
uv run llm init                 # write a starter config
uv run llm                      # pick a model, then chat
uv run llm chat -mkimi-k3       # skip the picker
uv run llm chat --web           # with web search + page fetching
uv run llm chat -mgpt-5 -s "answer in one sentence" -t0.2
uv run llm ask "explain b-trees" -mdeepseek-chat
cat main.py | uv run llm ask "review this" -mclaude-sonnet-5
uv run llm ask --markdown "compare X and Y"   # render markdown

```

In chat: `/model` (reopens the picker, keeps your history), `/system`,
`/clear`, `/history`, `/save`, `/web`, `/markdown`, `/temp`, `/providers`,
`/exit`. Enter sends; alt+enter inserts a newline.

## Config

`llm init` writes `$XDG_CONFIG_HOME/llm/config.toml`, falling back to
`~/.config/llm/config.toml` when `$XDG_CONFIG_HOME` is unset (override the
path entirely with `$LLM_CONFIG`):

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

Aliases appear at the top of the picker. Custom providers behave exactly like the built-ins, including live model listing.

Built in:

- openai
- anthropic
- google
- moonshot
- deepseek
- xai
- zai
- mistral
- groq
- cerebras
- together
- fireworks
- deepinfra
- perplexity
- openrouter
- ollama
- lmstudio

The two local ones need no key.

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

`ChatSession` keeps the message list, tracks token usage, runs the tool loop, and tags every streamed fragment by kind:

- `text`
- `reasoning`
- `tool_call`
- `tool_result`
- `notice`

Reasoning and tool chatter never leak into the answer. Tools are just a schema plus a callable, so your own are a few lines:

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

`ToolBox.call` never raises a failing tool returns its error as text so the
model can recover instead of the turn dying.

## Development

```shell
uv format
uv run ruff check
uv run pyrefly check
uv run pytest
```
