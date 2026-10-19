# Chat GUI for `llm` — design

## Intent

Krish uses `llm chat` (a terminal REPL over any OpenAI-compatible API, with
optional web search / page fetching tools) every day. The terminal is fine for
sending prompts but poor for *reading* and *returning to* conversations. This
adds a local browser front end to the same engine, fixing three concrete
annoyances from a real session (a `--web` chat with Kimi K2.6 about r/foodnyc
restaurants):

1. Thinking text is long and cannot be collapsed once it has scrolled past.
2. Chats vanish on exit; there is no way to list or resume old ones.
3. The model's markdown prints raw (`**bold**`, `### headings`).

Success: opening the GUI is as quick as opening the REPL, and old chats are
one click away and resumable.

## What it is

A Python web server on the user's own Mac, opened in Safari/Chrome. Not a
packaged app, not hosted, not shared. It reuses `ChatSession`, `Registry`,
`catalog`, and the web toolbox unchanged in behaviour, so models, API-key
inference, Tavily/Brave search, Reddit fetching, and the date preamble all work
exactly as in the terminal.

## Goals

- **Layout:** sidebar of past chats, grouped Today / Yesterday / Last week /
  Older, with a "New chat" button. Main pane: header (model, web-access
  toggle), message list, input box. Settled during brainstorming against a
  mockup (layout "A"); light/dark follows the system.
- **Thinking blocks:** each reasoning stream is a collapsible block showing its
  duration ("Thinking · 22s"). Open while streaming, collapse automatically
  when the answer begins, click to toggle. A keyboard shortcut and a button
  expand/collapse *all* blocks in the chat.
- **Tool calls:** one quiet line each, like the mockup (`web_search …`,
  `fetch_url …` and a short result summary), matching the `Delta`
  `tool_call`/`tool_result` events. Not collapsible in v1.
- **Rendered markdown:** headings, bold/italic, lists, links, tables, code
  blocks, rendered live as tokens stream.
- **History:** every chat is saved automatically; clicking one in the sidebar
  reopens it and the user can keep going. Delete a chat from the sidebar.
- **Controls in the page** for what the REPL does with slash commands: switch
  model (the existing picker data, `catalog.list_models`, per provider), toggle
  web tools, set the system prompt, set temperature.
- **Stop button** to abort a reply in progress (the REPL's ctrl-c).
- Starts with one command and opens the browser.

## Non-goals

No accounts, login, hosting, sharing, or multi-user. No mobile layout. No
packaged `.app`. No syntax highlighting, code-copy buttons, search across chats,
chat renaming, export UI, or file/image attachments in v1. No shared history
with the terminal REPL, which is left completely unchanged.

## Anti-goals (failure even if it technically works)

- Slower to reach a prompt than the terminal.
- Losing a chat because of a crash or ctrl-c mid-reply, or a corrupt file that
  hides every other chat.
- A page, or a web page the model fetched, being able to make the browser run
  injected HTML/JS, or any other website being able to drive the local server.
- API keys ever reaching the browser.

## Decisions made with Krish

- **Stack:** Litestar app served by Granian (ASGI). Streaming to the browser
  over server-sent events using Litestar's SSE support. Robyn and Starlette were
  considered; Litestar chosen once the Python bump below made its wheels
  available.
- **Python:** lower `requires-python` from `>=3.15` to `>=3.14`. Reason: on
  3.15, Litestar needs `pyyaml`, which has no 3.15 wheels, forcing a source
  build that the user's strict `cc` wrapper (`-Werror`) can break. Nothing in
  the code needs 3.15 (all of `src/` and `tests/` parse under 3.14.8; the tests
  have not yet been run on 3.14). Regenerate `uv.lock`.
- **No Node, no build step.** Frontend is plain HTML/CSS/JS served as static
  files.
- **Markdown:** `marked` for parsing plus `DOMPurify` for sanitising its output
  (model text can include fetched web content). Re-render the growing message
  on `requestAnimationFrame` rather than per token. Both libraries are
  committed into the repo as pinned copies, not loaded from a CDN, so the page
  works offline and cannot change under us.
- **Storage:** JSON Lines, one file per chat, no database.
- **Terminal REPL unchanged.**
- **Replies survive switching chats.** A reply keeps running and saving when the
  user clicks another chat; several chats may stream at once, and the sidebar
  marks busy ones. Stop, or the page going away, cancels a reply and saves what
  arrived so far (so no tokens are spent with nobody watching).

## Calls made by the document author

Krish did not specify these; the author chose them. Build them as written,
not as open questions. Krish reviews the list on handover and may reverse any.

- **Launch:** a new subcommand `llm gui` (options `--model`, `--port`,
  `--no-open`), rather than a flag on `chat`, because `chat --web` already means
  "web search tools".
- **Storage location:** `$XDG_DATA_HOME/llm/chats/<id>.jsonl` (default
  `~/.local/share`), as chats are data, not cache (the existing history and
  model-list caches use `~/.cache/llm/`).
- **File format:** every line is a JSON object with a `type`. The first is
  `meta` (id, created time). `settings` lines hold model spec, system prompt,
  temperature, and web on/off; one is written at creation and another each time
  the user changes any of them, and the latest wins. `message` lines hold one
  `Message` each (`role`, `content`, `reasoning`, `reasoning_seconds`,
  `tool_calls`, `tool_call_id`). The file is strictly append-only, so
  mid-chat setting changes never rewrite earlier lines, and a truncated final
  line is ignored on load. A resumed chat keeps its last saved settings.
- **Write strategy and crash safety:** the user message is appended *before*
  the API call. `ChatSession` already adds each assistant/tool message as a
  round finishes (even on an abandoned stream), so append those to the file as
  they appear. Stop, disconnect, or a server crash therefore leaves everything
  completed so far. On load, drop a trailing assistant message whose
  `tool_calls` lack matching `tool` messages (and any partial tool run after
  it), since the API rejects that shape; text already shown is kept.
- **Core-type changes:** `Message` gains `reasoning: str = ""` and
  `reasoning_seconds: float = 0.0`; `ChatSession._round` fills them from the
  reasoning stream (timing first reasoning delta to first non-reasoning one).
  `as_param` is unchanged, so API payloads and the REPL behave as before.
  `ChatSession` stays the authority for system prompt, `params`, and `tools`;
  the GUI layer mutates those attributes exactly as the REPL commands do.
- **Python 3.14 gate:** do the version bump and regenerate `uv.lock` first, and
  run the full existing test suite on 3.14 before anything else. Fix trivial
  breakage (e.g. a stdlib difference). If something non-trivial fails, or a
  dependency lacks 3.14 wheels, stop and report rather than redesigning.
- **Web toggle:** always available, because the keyless DuckDuckGo backend is
  the fallback. Show the active backend (`ui.search_status()`) next to the
  toggle. The toolbox is built per chat (`tools.web_toolbox()`).
- **Auth details:** the token defends against other websites, not other local
  processes or users on the machine. Send messages with `fetch` streaming over
  POST (not `EventSource`), so no GET endpoint triggers work. Cookie is
  `SameSite=Strict`, `HttpOnly`. A request with a bad or missing token gets a
  plain 403 page saying to relaunch `llm gui`.
- **Reasoning is stored.** Add an optional `reasoning` string (and its
  duration) to `Message`; it is persisted and shown but never sent back to the
  API (`Message.as_param` is unchanged).
- **Sidebar title:** the first user message, trimmed to about 60 characters. No
  model call.
- **Sidebar order:** most recently updated first.
- **Security of the local server:** bind to `127.0.0.1` only; a random
  per-launch token carried in the opening URL (stored in a cookie afterwards)
  required on every request, plus an `Origin`/`Host` check. This stops other
  websites in the browser (which can reach `localhost`) from spending tokens or
  triggering `fetch_url`.
- **Events over SSE:** one JSON event per `Delta` (`text`, `reasoning`,
  `tool_call`, `tool_result`, `notice`), plus `usage`, `error`, and `done`. A
  client disconnect or Stop closes the generator (as `repl._turn_streamed` does
  with `closing(...)`), and whatever was produced is saved.
- **Concurrency:** one active reply per chat, many chats at once.
  `ChatSession.stream` is a blocking generator over the sync OpenAI client; run
  each reply in a worker thread, off the event loop, as a server-side task that
  is *not* tied to the HTTP request carrying its events. Buffer its events per
  chat so the browser can reattach (replay the buffer, then continue live) when
  the user returns to a chat that is still running. Cancel a reply on Stop, or
  when the page itself is gone (heartbeat lost after a short grace period so a
  reload doesn't kill it), closing the generator so partial output is saved.
  Switching chats inside the page must not cancel anything.

## Constraints

- Python 3.14+, managed with `uv`; deps added via `uv add`. If a dependency must
  compile native code, pin the compiler (`CC=/usr/bin/cc CXX=/usr/bin/c++`);
  never alter the `~/.local/bin/cc` wrapper.
- Match the codebase: type-annotated, `from __future__ import annotations`,
  small modules with docstrings, ruff (line length 88) and pyrefly clean,
  pytest tests that use the fakes in `tests/conftest.py`.
- Existing behaviour of `llm chat`, `ask`, and the rest must not change; the
  existing tests must keep passing under 3.14.
- Providers and API keys are read from the environment on the server only.

## Left to the builder

Package layout (suggestion: `src/llm/gui/` holding the Litestar app, a chat
store, and a `static/` directory), exact route shapes, JS structure and CSS,
how chats are held in memory between requests, file-id scheme, exact
shortcut keys, port default, and how unknown/legacy fields in a chat file are
handled.

## Testing expectations

Chat store round-trips (including a truncated last line); `Delta` → SSE event
mapping; API tests for create / list / resume / delete / send using the existing
fake client, including the token and Origin rejection paths; reasoning
persisted but absent from the API payload. Browser behaviour (thinking toggle,
streaming render, resume) verified by actually running the GUI and looking at
it.
