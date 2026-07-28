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

Every provider reads one environment variable. Put them in `.env` (loaded
automatically, git-ignored) or export them:

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

## Commands

```sh
llm                       # pick a model, then chat
llm chat -m kimi-k3       # skip the picker
llm chat -m gpt-5 -s "answer in one sentence" -t 0.2
llm ask "explain b-trees" -m deepseek-chat
cat main.py | llm ask "review this" -m claude-sonnet-5
llm ask --markdown "compare X and Y"   # buffer, then render markdown
llm providers
llm init                  # write a starter config
```

In chat: `/model` (reopens the picker, keeps your history), `/system`,
`/clear`, `/history`, `/save`, `/markdown`, `/temp`, `/providers`, `/exit`.
Enter sends; alt+enter inserts a newline.

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

registry = Registry.load()
ref, client = registry.open("kimi-k3")  # a plain openai.OpenAI client

session = ChatSession(ref=ref, client=client, system="be terse")
for delta in session.stream("hello"):
    if delta.kind == "text":
        print(delta.text, end="")
```

`ChatSession` keeps the message list, tracks token usage, and separates
streamed reasoning (`delta.kind == "reasoning"`) from the answer so it never
leaks back into the conversation history.

## Development

```sh
uv run pytest
uv run ruff check . && uv run ruff format .
uv run pyrefly check
```
