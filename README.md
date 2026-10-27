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

Three tools are offered: `web_search`, `fetch_url` and `find_in_page`.
`/web` toggles them mid-conversation.

Long pages come a part at a time: 6,000 characters, or 25,000 for a Reddit
thread, cut at a line break and ending with the offset to read on from.
`find_in_page` searches a page for words or a phrase ("vegetarian", "eggplant
parm") and returns the matching passages with their offsets, best matches
first, so the model can jump to the part of a long menu or thread it needs.
Each conversation keeps the pages it has read for an hour, so reading on or
searching costs no new request.

A turn may make up to 40 tool calls (`TOOL_BUDGET` in `src/llm/chat.py`),
enough to follow a thread out to each place it names. As the budget runs low,
tool results remind the model to wrap up; once it is spent, the model gets one
more round with tools switched off, so the turn still ends in an answer.

When a fetch fails, the model is told what kind of failure it was and what to
try instead, rather than handed a raw HTTP error. Bot checks (Cloudflare,
DataDome, Akamai and the like) count as blocks even when they arrive with a
200, and pages that only render with JavaScript are reported as such. The
tools also remember what they've run: a call that already failed isn't run
again (except one retry for timeouts and server errors), a repeat of one that
succeeded points back to the earlier result, and a site that has blocked two
requests isn't fetched again that conversation.

Models also drift over a long run, so the loop works around the usual ways:

- **Tool calls written as text.** A reply that writes its call out (as JSON at
  the end, in a code fence, as `fetch_url({...})`, or as Kimi, DeepSeek or Qwen
  tool-call tokens leaking through) is rewritten in history as a real call and
  run. If the attempt can't be read, the model is asked once to call the tool
  properly instead of the turn ending on it.
- **Damaged arguments** (fenced, wrapped in prose, trailing commas,
  single-quoted, JSON inside a string) are repaired, and a tool name like
  `functions.fetch_url` is matched to `fetch_url`. Errors that can't be
  repaired show the tool's parameters and an example call, and an unknown tool
  suggests the closest real one.
- **No progress.** After three failed calls in a row, the result says so,
  lists them, and tells the model to change approach or move on.
- **Fading instructions.** Every eight calls, a tool result restates the few
  rules that matter most, near the end of the context where the model is
  looking.

A turn where any of that happened ends with a count, e.g. `harness: 2 tool
calls recovered from text, 1 no-progress warning`, so you can see how much a
model leaned on it. `session.last_events` has the same counts.

Pages are requested with a browser's headers (a current Chrome User-Agent,
`Accept`, `Accept-Language`), since many small sites turn away anything else.
That won't get past serious bot checks, which come back as blocks as above.
APIs and Reddit's feeds still get the descriptive `llm-cli` agent.

Old tool results are shortened in what's sent to the model. Counting back from
the newest, about 120,000 characters of tool output (`RECENT_TOOL_CHARS`) go
whole; older results over 2,000 characters become a one-line stub naming the
call. The saved history keeps everything. Making the same call again returns
the full result from memory without a new request, and the preamble tells the
model to keep brief notes as it goes so findings outlive the pages.

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
uv run llm fetch https://… --offset 6000        # the next part
uv run llm fetch https://… --find vegetarian    # what find_in_page returns
LLM_SEARCH_BACKEND=exa    # pin one explicitly
```

`fetch_url` resolves the host and refuses anything that isn't a public
address. Loopback, private ranges, and link-local (including the cloud
metadata endpoint) are blocked, and every redirect hop is re-checked. The
model picks those URLs, often from a page it just read, so a prompt injected into that page shouldn't be able to aim the fetcher at your network. Responses are capped at 2 MB and pages are shown 6,000 characters at a time.

**Reddit** turns away scripted requests (a JavaScript challenge, a login wall,
or a 403), so Reddit links take their own route. With `LLM_REDDIT_URL` pointing
at a self-hosted [Redlib](https://github.com/redlib-org/redlib), threads come
back in full as a comment tree, 25,000 characters at a time. Without it, or if
Redlib is down, they fall back to Reddit's RSS feeds, which allow about one page
a minute and carry only some comments. `llm web` shows which route is active.

Redlib's crates.io release is stale and gets 403s; build from `main`
(needs `cmake`, e.g. `uv tool install cmake`):

```shell
cargo install --git https://github.com/redlib-org/redlib --locked
echo 'LLM_REDDIT_URL=http://127.0.0.1:8089' >> .env
```

You don't need to run it yourself. When `LLM_REDDIT_URL` is a loopback address
with a port, nothing is answering there, and `redlib` is on your `PATH`, the
first Reddit link of a session starts it (about 2 seconds) and it's stopped
when the session ends. If it's already running, say from another session or
a service you've set up, that one is used instead.

Redlib reads Reddit by presenting itself as Reddit's official app, which is
against Reddit's terms. No account is involved, but expect it to break
occasionally when Reddit changes something.

You'll know it has broken when Reddit links come back as "Redlib could not load
…" errors, or keep arriving with the "Redlib … is unreachable" note (it failed
to start). The fix is to rebuild from `main`, which usually has a patch
within a few days:

```shell
cargo install --git https://github.com/redlib-org/redlib --locked --force
pkill redlib    # stop any old copy still running; the next session starts the new one
```

If that doesn't help yet, check the
[Redlib issues](https://github.com/redlib-org/redlib/issues). Until it's fixed,
Reddit links fall back to the feeds on their own.

**Thinking models** that need their earlier reasoning back get it, as
`reasoning_content` on the assistant messages they wrote. How much depends on
the model, since it's billed as input: Kimi K3, K3.1 Pro and K2.7 Code require
it on every assistant message in the conversation; other Kimi models only
within the current turn's tool calls. Everything else gets none. The rules are
`reasoning_replay` on each provider in `src/llm/registry.py`.

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
Typed input is remembered across sessions in `$XDG_STATE_HOME/llm/history`
(`~/.local/state/llm/history` by default).

Every request starts with a short built-in preamble giving today's date (and,
with `/web` on, guidance on using the search tools). A `--system`/`/system`
prompt is appended after it rather than replacing it. The text lives in
`PREAMBLE`/`WEB_PREAMBLE` in `src/llm/chat.py`.

## Chat GUI

```shell
uv run llm gui                       # serves http://127.0.0.1:8765 and opens it
uv run llm gui -mkimi-k3 --port 9000 --no-open
```

The same engine as `llm chat` (models, web search, Reddit, the date preamble)
in a browser tab, for the parts a terminal is bad at:

- **Past chats** live in a sidebar, grouped by day; click one to read it or keep going.
- **Thinking** is a collapsible block with its duration. It stays open while the
  model reasons and folds away when the answer starts. Click one to toggle it, or use
  the _Thinking_ button / <kbd>Alt</kbd>+<kbd>T</kbd> for all of them.
- **Markdown** is rendered as it streams, sanitised before it reaches the page
  (model output can include text from pages it fetched), with images and inline
  styles stripped and a content-security policy that forbids inline script.
- **Replies keep running** when you switch to another chat; the sidebar marks busy
  ones. _Stop_ cancels a reply, and so does closing the tab (after ~15 s), keeping
  what had arrived.
- **Settings** (model, system prompt, temperature, web on/off) are per chat, in the header.

Chats are saved as one JSON Lines file each in `$XDG_DATA_HOME/llm/chats/`
(`~/.local/share/llm/chats/` by default), appended to as the conversation
goes, so a crash costs at most the last line.

It listens on `127.0.0.1` only. The link `llm gui` prints carries a random
per-launch token; requests without it, from another origin, or for another
hostname are refused. That protects you from other websites, not from other
processes on your Mac. `llm chat` and `llm ask` are unaffected.

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

The two local ones (`ollama`, `lmstudio`) need no key.

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

The project pins Python 3.14 (`.python-version`): the GUI's dependencies
(Litestar needs `pyyaml`) don't all ship 3.15 wheels yet.

```shell
uv format
uv run ruff check
uv run pyrefly check
uv run pytest
```
