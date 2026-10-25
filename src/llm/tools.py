"""Tools the model can ask us to run.

A tool is a JSON schema the model sees plus a Python callable it never sees.
The model can only *request* a call; :meth:`ToolBox.call` is what actually
runs, and it returns errors as text so a bad call is recoverable rather than
fatal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import web

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["Tool", "ToolBox", "web_toolbox"]


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


class ToolBox:
    """The set of tools offered on a request, and the dispatcher for them."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools = {tool.name: tool for tool in tools}

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

    def call(self, name: str, arguments: str) -> str:
        """Run a requested tool. Never raises: failures come back as text."""
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: no tool named {name!r}. Available: {', '.join(self.names)}"
        try:
            kwargs = json.loads(arguments) if arguments.strip() else {}
        except ValueError as exc:
            return f"Error: arguments were not valid JSON ({exc})"
        if not isinstance(kwargs, dict):
            return "Error: arguments must be a JSON object"
        try:
            return tool.run(**kwargs)
        except TypeError as exc:
            return f"Error: wrong arguments for {name} ({exc})"
        except web.WebError as exc:
            return f"Error: {exc}"
        # ruff: ignore[blind-except] - a tool must never kill the turn
        except Exception as exc:
            return f"Error: {type(exc).__name__}: {exc}"


def _web_search(query: str, count: int = web.DEFAULT_RESULTS) -> str:
    results = web.search(query, count=max(1, min(count, 10)))
    return web.render(results)


def _fetch_url(url: str) -> str:
    return web.fetch(url)


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

FETCH_URL = Tool(
    name="fetch_url",
    description=(
        "Download a public http(s) URL and return its readable text, "
        "truncated if long. Use it to read a page found via web_search. "
        "Reddit links work too and return the post with its comment tree "
        "(scores in brackets, replies indented)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "An absolute http:// or https:// URL.",
            }
        },
        "required": ["url"],
    },
    run=_fetch_url,
)


def web_toolbox() -> ToolBox:
    """The default toolbox: search plus page fetching."""
    return ToolBox((WEB_SEARCH, FETCH_URL))
