"""Fetching and searching the web on the model's behalf.

The model never touches the network — it emits a tool call, this module does
the HTTP, and the result goes back as a ``role: "tool"`` message.

Search backends follow the same shape as providers in :mod:`llm.registry`:
one env var each, first ready backend wins, overridable with
``$LLM_SEARCH_BACKEND``.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "Result",
    "SearchBackend",
    "WebError",
    "active_backend",
    "backends",
    "fetch",
    "search",
]

USER_AGENT = "llm-cli/0.1 (+https://github.com/)"
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
TIMEOUT = 20.0
DEFAULT_PAGE_CHARS = 6000
DEFAULT_RESULTS = 5


class WebError(Exception):
    """A request could not be completed, or was refused by the guard."""


@dataclass(frozen=True, slots=True)
class Result:
    title: str
    url: str
    snippet: str = ""

    def render(self, index: int) -> str:
        head = f"[{index}] {self.title}\n    {self.url}"
        return f"{head}\n    {self.snippet}" if self.snippet else head


@dataclass(frozen=True, slots=True)
class SearchBackend:
    """A search API. ``env_var`` of ``None`` means no credential is needed."""

    name: str
    env_var: str | None
    label: str
    #: for self-hosted backends: the env var holding the instance URL
    url_var: str | None = None

    def credential(self) -> str | None:
        if self.env_var is None:
            return None
        return os.environ.get(self.env_var, "").strip() or None

    def instance_url(self) -> str | None:
        if self.url_var is None:
            return None
        return os.environ.get(self.url_var, "").strip() or None

    def is_ready(self) -> bool:
        if self.url_var is not None:
            return self.instance_url() is not None
        if self.env_var is None:
            return True  # the keyless fallback
        return self.credential() is not None


#: Priority order: a configured API beats the keyless scrape.
BACKENDS: tuple[SearchBackend, ...] = (
    SearchBackend("brave", "BRAVE_SEARCH_API_KEY", "Brave Search"),
    SearchBackend("tavily", "TAVILY_API_KEY", "Tavily"),
    SearchBackend("exa", "EXA_API_KEY", "Exa"),
    SearchBackend("searxng", None, "SearXNG (self-hosted)", url_var="SEARXNG_URL"),
    SearchBackend("duckduckgo", None, "DuckDuckGo (unofficial, no key)"),
)


def backends() -> tuple[SearchBackend, ...]:
    return BACKENDS


def active_backend() -> SearchBackend:
    """The backend that :func:`search` will use."""
    chosen = os.environ.get("LLM_SEARCH_BACKEND", "").strip().lower()
    if chosen:
        for backend in BACKENDS:
            if backend.name == chosen:
                return backend
        known = ", ".join(b.name for b in BACKENDS)
        msg = f"unknown LLM_SEARCH_BACKEND {chosen!r}; known backends: {known}"
        raise WebError(msg)
    return next(b for b in BACKENDS if b.is_ready())


def search(query: str, *, count: int = DEFAULT_RESULTS) -> list[Result]:
    """Run ``query`` against the active backend."""
    backend = active_backend()
    if not backend.is_ready():
        msg = f"{backend.label} is not configured ({backend.env_var} is unset)"
        raise WebError(msg)
    runner = _RUNNERS[backend.name]
    try:
        return runner(backend, query, count)[:count]
    except httpx.HTTPError as exc:
        msg = f"{backend.label} request failed: {exc}"
        raise WebError(msg) from exc


def fetch(url: str, *, max_chars: int = DEFAULT_PAGE_CHARS) -> str:
    """Download ``url`` and return it as readable plain text."""
    response = _get_guarded(url)
    content_type = response.headers.get("content-type", "")
    body = response.text
    text = body if "html" not in content_type.lower() else html_to_text(body)
    text = text.strip()
    if not text:
        return "(the page returned no readable text)"
    if len(text) > max_chars:
        return f"{text[:max_chars]}\n\n[truncated at {max_chars} characters]"
    return text


# -- HTTP with a guard ------------------------------------------------------


def _get_guarded(url: str) -> httpx.Response:
    """GET ``url``, re-checking every redirect hop against the address guard."""
    current = url
    with httpx.Client(
        timeout=TIMEOUT,
        follow_redirects=False,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            _assert_public(current)
            response = client.get(current)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    break
                current = urljoin(current, location)
                continue
            _assert_small_enough(response)
            response.raise_for_status()
            return response
    msg = f"too many redirects starting at {url}"
    raise WebError(msg)


def _assert_public(url: str) -> None:
    """Refuse anything that is not a public http(s) address.

    The model chooses these URLs, so this keeps a prompt from steering the
    fetcher at localhost or the private network.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        msg = f"only http(s) URLs can be fetched, got {parsed.scheme or url!r}"
        raise WebError(msg)
    host = parsed.hostname
    if not host:
        msg = f"no host in URL {url!r}"
        raise WebError(msg)
    try:
        infos = socket.getaddrinfo(host, parsed.port or 0, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        msg = f"could not resolve {host}: {exc}"
        raise WebError(msg) from exc
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global or address.is_multicast:
            msg = f"refusing to fetch {host} ({address}): not a public address"
            raise WebError(msg)


def _assert_small_enough(response: httpx.Response) -> None:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        msg = f"response is {declared} bytes, over the {MAX_BYTES} limit"
        raise WebError(msg)
    if len(response.content) > MAX_BYTES:
        msg = f"response exceeded the {MAX_BYTES} byte limit"
        raise WebError(msg)


# The backend endpoints below are configured by the user, not chosen by the
# model, so they skip the address guard — a self-hosted SearXNG on localhost is
# a perfectly legitimate target.


def _post_json(
    url: str, payload: Mapping[str, Any], headers: Mapping[str, str]
) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.post(
            url, json=dict(payload), headers={"User-Agent": USER_AGENT, **headers}
        )
        response.raise_for_status()
        return response.json()


def _get_json(
    url: str, params: Mapping[str, Any], headers: Mapping[str, str]
) -> dict[str, Any]:
    with httpx.Client(timeout=TIMEOUT) as client:
        response = client.get(
            url, params=dict(params), headers={"User-Agent": USER_AGENT, **headers}
        )
        response.raise_for_status()
        return response.json()


# -- backends ---------------------------------------------------------------


def _brave(backend: SearchBackend, query: str, count: int) -> list[Result]:
    data = _get_json(
        "https://api.search.brave.com/res/v1/web/search",
        {"q": query, "count": count},
        {
            "Accept": "application/json",
            "X-Subscription-Token": backend.credential() or "",
        },
    )
    results = data.get("web", {}).get("results", [])
    return [
        Result(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=_clean(item.get("description", "")),
        )
        for item in results
    ]


def _tavily(backend: SearchBackend, query: str, count: int) -> list[Result]:
    data = _post_json(
        "https://api.tavily.com/search",
        {"query": query, "max_results": count},
        {"Authorization": f"Bearer {backend.credential()}"},
    )
    return [
        Result(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=_clean(item.get("content", "")),
        )
        for item in data.get("results", [])
    ]


def _exa(backend: SearchBackend, query: str, count: int) -> list[Result]:
    data = _post_json(
        "https://api.exa.ai/search",
        {"query": query, "numResults": count, "contents": {"text": True}},
        {"x-api-key": backend.credential() or ""},
    )
    return [
        Result(
            title=item.get("title") or item.get("url", ""),
            url=item.get("url", ""),
            snippet=_clean(item.get("text", ""))[:300],
        )
        for item in data.get("results", [])
    ]


def _searxng(backend: SearchBackend, query: str, count: int) -> list[Result]:
    base = (backend.instance_url() or "").rstrip("/")
    data = _get_json(
        f"{base}/search", {"q": query, "format": "json", "count": count}, {}
    )
    return [
        Result(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=_clean(item.get("content", "")),
        )
        for item in data.get("results", [])
    ]


def _duckduckgo(_backend: SearchBackend, query: str, _count: int) -> list[Result]:
    """Scrape the keyless HTML endpoint. Unofficial, so it may break."""
    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        response = client.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
    parser = _DuckDuckGoParser()
    parser.feed(response.text)
    return parser.results()


_RUNNERS: Mapping[str, Any] = {
    "brave": _brave,
    "tavily": _tavily,
    "exa": _exa,
    "searxng": _searxng,
    "duckduckgo": _duckduckgo,
}


# -- HTML ------------------------------------------------------------------

_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "head"})
_BREAK_TAGS = frozenset({
    "p", "br", "div", "section", "article", "li", "tr",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre",
})  # fmt: skip


class _TextExtractor(HTMLParser):
    """Turn HTML into readable text without pulling in a parser dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: object) -> None:
        del attrs
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BREAK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BREAK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return _collapse("".join(self._parts))


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._links: list[tuple[str, str]] = []
        self._snippets: list[str] = []
        self._in_link = False
        self._in_snippet = False
        self._href = ""
        self._title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = (values.get("class") or "").split()
        if tag == "a" and "result-link" in classes:
            self._in_link = True
            self._href = values.get("href") or ""
            self._title = []
        elif tag == "td" and "result-snippet" in classes:
            self._in_snippet = True
            self._snippets.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            self._links.append((_collapse("".join(self._title)), self._href))
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._title.append(data)
        elif self._in_snippet and self._snippets:
            self._snippets[-1] += data

    def results(self) -> list[Result]:
        return [
            Result(title=_clean(title), url=url, snippet=_clean(snippet))
            for (title, url), snippet in zip(
                self._links, _padded(self._snippets, len(self._links)), strict=False
            )
            if url.startswith("http")
        ]


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def _padded(values: list[str], size: int) -> Iterator[str]:
    for index in range(size):
        yield values[index] if index < len(values) else ""


def _collapse(text: str) -> str:
    lines = (" ".join(line.split()) for line in text.splitlines())
    kept = [line for line in lines if line]
    return "\n".join(kept)


def _clean(text: str) -> str:
    return _collapse(html_to_text(text)).replace("\n", " ").strip()


def render(results: list[Result]) -> str:
    """Format results for a tool message."""
    if not results:
        return "No results."
    return "\n\n".join(
        result.render(index) for index, result in enumerate(results, start=1)
    )


def dumps(results: list[Result]) -> str:
    return json.dumps(
        [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
    )
