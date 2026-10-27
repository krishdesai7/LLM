"""Tools the model can ask us to run.

A tool is a JSON schema the model sees plus a Python callable it never sees.
The model can only *request* a call; :meth:`ToolBox.call` is what actually
runs, and it returns errors as text so a bad call is recoverable rather than
fatal.

A box also remembers what it has run. Models that hit a wall (a 403, a bot
check) tend to try the same call again rather than change course, so a repeat
of a failed call, or a fetch from a site that keeps blocking, is answered with
the earlier failure instead of being run. A repeat of one that worked gets the
earlier result back, which is how a model recovers a result that
:mod:`llm.chat` has since shortened.

It also helps a model that drifts: damaged arguments are repaired (see
:mod:`llm.repair`), errors show how the tool should have been called, and a
run of failures earns a nudge to change approach. ``events`` counts each of
these per turn, so you can see how much a model leaned on them.
"""

from __future__ import annotations

import difflib
import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from . import repair, web

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "Tool",
    "ToolBox",
    "fetch_url_tool",
    "find_in_page_tool",
    "web_toolbox",
]

#: Blocked fetches from one site before the box stops trying that site at all.
HOST_BLOCK_LIMIT: Final[int] = 2
#: Runs of one call that failed in a way worth retrying (a timeout, a 502).
MAX_TRIES: Final[int] = 2
#: Failed calls in a row before the model is told to change approach.
STALL_AFTER: Final[int] = 3


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[..., str]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def usage(self) -> str:
        """How to call this tool, for error messages: parameters and an example."""
        properties: dict[str, Any] = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        lines = [f"{self.name} takes a JSON object with:"]
        for name, spec in properties.items():
            kind = spec.get("type", "any")
            flags = f"{kind}, required" if name in required else kind
            about = f": {spec['description']}" if spec.get("description") else ""
            lines.append(f'  "{name}" ({flags}){about}')
        if len(lines) == 1:
            lines = [f"{self.name} takes an empty JSON object, {{}}."]
        example = {
            name: _example(name, properties[name].get("type"))
            for name in properties
            if name in required
        }
        lines.append(f"Example: {json.dumps(example)}")
        return "\n".join(lines)


def _example(name: str, kind: str | None) -> Any:
    match kind:
        case "integer":
            return 5
        case "number":
            return 1.5
        case "boolean":
            return True
    if "url" in name:
        return "https://example.com/page"
    return "search terms" if "query" in name else "text"


@dataclass(slots=True)
class _Attempt:
    """How a call went the last time it ran."""

    number: int
    output: str
    failed: bool
    retryable: bool = False
    tries: int = 1

    @property
    def may_retry(self) -> bool:
        return self.failed and self.retryable and self.tries < MAX_TRIES

    def repeat(self) -> str:
        if not self.failed:
            # In full, since the original may have been shortened since.
            return (
                f"(This is the same call as #{self.number}, so no new request "
                f"was made. Its result again:)\n\n{self.output}"
            )
        failed = "failed twice" if self.tries > 1 else "failed"
        return (
            f"Error: not run, because it repeats call #{self.number}, which "
            f"{failed} and would fail the same way again:\n\n{self.output}"
        )


class ToolBox:
    """The set of tools offered on a request, and the dispatcher for them."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._seen: dict[str, _Attempt] = {}
        self._blocks: dict[str, list[int]] = {}
        self._failures: list[str] = []
        self._count = 0
        #: what the box did to keep the model on track, this turn
        self.events: Counter[str] = Counter()

    def __bool__(self) -> bool:
        return bool(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    @property
    def tools(self) -> list[Tool]:
        """The tools themselves, for building a box on top of this one."""
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def reset(self) -> None:
        """Forget earlier calls, as when the conversation is cleared."""
        self._seen.clear()
        self._blocks.clear()
        self._count = 0
        self.start_turn()

    def start_turn(self) -> None:
        """Begin counting afresh. What was tried stays remembered."""
        self._failures.clear()
        self.events.clear()

    def call(self, name: str, arguments: str) -> str:
        """Run a requested tool. Never raises: failures come back as text."""
        self._count += 1
        output = self._dispatch(name, arguments)
        if not output.startswith("Error:"):
            self._failures.clear()
            return output
        reason = _compact(output.removeprefix("Error:").split("\n", 1)[0], 100)
        self._failures.append(
            f"#{self._count} {name}({_compact(arguments, 60)}): {reason}"
        )
        if len(self._failures) % STALL_AFTER:
            return output
        self.events["stalled"] += 1
        tried = "\n".join(f"- {line}" for line in self._failures[-STALL_AFTER:])
        return (
            f"{output}\n\n[No progress: the last {STALL_AFTER} calls all failed:\n"
            f"{tried}\nVariations on an approach that keeps failing won't start "
            "working. Change tack: try a different kind of source (another site "
            "that covers this, a listing or review page, a search from a "
            "different angle), or record this as a dead end and move on to "
            "what's left.]"
        )

    def _dispatch(self, name: str, arguments: str) -> str:
        known = repair.canonical_name(name, self._tools)
        if known is None:
            return self._unknown(name)
        if known != name:
            self.events["repaired"] += 1
        tool = self._tools[known]
        try:
            kwargs, repaired = repair.parse_arguments(arguments)
        except ValueError as exc:
            if "JSON object" in str(exc):
                return f"Error: arguments must be a JSON object.\n{tool.usage()}"
            return f"Error: arguments were not valid JSON ({exc}).\n{tool.usage()}"
        if repaired:
            self.events["repaired"] += 1
        key = _call_key(known, kwargs)
        earlier = self._seen.get(key)
        if earlier is not None and not earlier.may_retry:
            self.events["refused"] += 1
            return earlier.repeat()
        url = kwargs.get("url")
        host = web.host_of(url) if isinstance(url, str) else None
        if host and len(blocks := self._blocks.get(host, [])) >= HOST_BLOCK_LIMIT:
            self.events["refused"] += 1
            numbers = ", ".join(f"#{n}" for n in blocks)
            return (
                f"Error: not run. {host} blocked calls {numbers}, so it will "
                "block this one too. Get this information from a different site."
            )
        output, error = _run(tool, kwargs)
        self._seen[key] = _Attempt(
            number=self._count,
            output=output,
            failed=output.startswith("Error:"),
            retryable=isinstance(error, web.FetchError) and error.retryable,
            tries=earlier.tries + 1 if earlier else 1,
        )
        if host and isinstance(error, web.FetchError) and error.kind == "blocked":
            self._blocks.setdefault(host, []).append(self._count)
        return output

    def _unknown(self, name: str) -> str:
        close = difflib.get_close_matches(name, self.names, n=1, cutoff=0.5)
        if close:
            return (
                f"Error: no tool named {name!r}. Did you mean {close[0]!r}?\n"
                f"{self._tools[close[0]].usage()}"
            )
        return f"Error: no tool named {name!r}. Available: {', '.join(self.names)}"


def _run(tool: Tool, kwargs: dict[str, Any]) -> tuple[str, Exception | None]:
    """The tool's output, or an error message and the exception behind it."""
    try:
        return tool.run(**kwargs), None
    except TypeError as exc:
        return f"Error: wrong arguments for {tool.name} ({exc}).\n{tool.usage()}", exc
    except web.WebError as exc:
        return f"Error: {exc}", exc
    # ruff: ignore[blind-except] - a tool must never kill the turn
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}", exc


def _call_key(name: str, kwargs: dict[str, Any]) -> str:
    """What makes two calls "the same", ignoring differences that change nothing.

    Spacing never matters; nor does case, except in URLs, whose paths can be
    case-sensitive. A URL's fragment and trailing slash are dropped too.
    """
    return json.dumps(
        [name, {key: _normalized(value) for key, value in kwargs.items()}],
        sort_keys=True,
        default=str,
    )


def _normalized(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = " ".join(value.split())
    if "://" in text:
        return text.partition("#")[0].rstrip("/")
    return text.casefold()


def _web_search(query: str, count: int = web.DEFAULT_RESULTS) -> str:
    results = web.search(query, count=max(1, min(count, 10)))
    return web.render(results)


WEB_SEARCH = Tool(
    name="web_search",
    description=(
        "Search the web and return ranked results with titles, URLs, and "
        "snippets. Use it for anything current, niche, or that you would "
        "otherwise be guessing at. Follow up with fetch_url to read a result "
        "in full."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query, phrased as you would type it.",
            },
            "count": {
                "type": "integer",
                "description": "How many results to return (1-10, default 5).",
            },
        },
        "required": ["query"],
    },
    run=_web_search,
)


def fetch_url_tool(pages: web.Pages) -> Tool:
    """``fetch_url``, reading through ``pages`` so long pages can be paged."""

    def run(url: str, offset: int = 0) -> str:
        return web.window(pages.text(url), offset, web.page_chars(url))

    return Tool(
        name="fetch_url",
        description=(
            "Download a public http(s) URL and return its readable text, a "
            "part at a time: a long page ends by giving the offset to read on "
            "from. Use it to read a page found via web_search. Reddit links "
            "work too and return the post with its comment tree (scores in "
            "brackets, replies indented)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "An absolute http:// or https:// URL.",
                },
                "offset": {
                    "type": "integer",
                    "description": (
                        "Character to start reading from, as given at the end "
                        "of an earlier fetch_url or by find_in_page (default 0)."
                    ),
                },
            },
            "required": ["url"],
        },
        run=run,
    )


def find_in_page_tool(pages: web.Pages) -> Tool:
    """``find_in_page``, searching the same ``pages`` that ``fetch_url`` reads."""

    def run(url: str, query: str) -> str:
        return web.find_passages(pages.text(url), query)

    return Tool(
        name="find_in_page",
        description=(
            "Search a page for words or a phrase and return the matching "
            "passages with their offsets, without reading all of it. Use it on "
            "long pages (a full menu, a big thread) to find what you need, then "
            "read around a match with fetch_url and its offset. Fetches the "
            "page first if it hasn't been read yet."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "An absolute http:// or https:// URL.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Words or a phrase to look for, e.g. 'vegetarian' or "
                        "'eggplant parm'. Ignores case; passages with more of "
                        "the words come first."
                    ),
                },
            },
            "required": ["url", "query"],
        },
        run=run,
    )


def web_toolbox() -> ToolBox:
    """The default toolbox: search, plus reading and searching pages.

    Each box has its own page cache, so one conversation's pages are reused
    by it alone.
    """
    pages = web.Pages()
    return ToolBox((WEB_SEARCH, fetch_url_tool(pages), find_in_page_tool(pages)))


def _compact(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}…"
