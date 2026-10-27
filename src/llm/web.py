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
import operator
import os
import re
import socket
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urljoin, urlparse

import httpx

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "FetchError",
    "Pages",
    "Result",
    "SearchBackend",
    "WebError",
    "active_backend",
    "backends",
    "fetch",
    "find_passages",
    "get_guarded",
    "host_of",
    "page_chars",
    "read",
    "search",
    "truncate",
    "window",
]

#: Identifies us to APIs: search backends, Redlib, Reddit's feeds.
USER_AGENT = "llm-cli/0.1 (+https://github.com/)"

#: Sent when fetching pages for the model. Plenty of small sites (a
#: restaurant's own page on shared hosting, say) turn away anything that does
#: not look like a browser. Keep the version near current Chrome, as a stale
#: one is itself a bot signal. Serious bot checks look at far more than
#: headers and still say no; those come back as BLOCKED.
BROWSER_HEADERS: Mapping[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
    ),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
}
MAX_BYTES = 2_000_000
MAX_REDIRECTS = 3
TIMEOUT = 20.0
DEFAULT_PAGE_CHARS = 6000
DEFAULT_RESULTS = 5


class WebError(Exception):
    """A request could not be completed, or was refused by the guard."""


FetchKind = Literal[
    "blocked",
    "rate_limited",
    "not_found",
    "rejected",
    "server_error",
    "timeout",
    "unreachable",
    "needs_js",
]

#: What the model should do next, per kind of failure. A bare "403 Forbidden"
#: tells it what went wrong but not that retrying is pointless, so models tend
#: to try the same URL again and again; saying so lets them switch sources.
_ADVICE: dict[FetchKind, str] = {
    "blocked": (
        "{host} refuses automated requests, so fetching it again will fail the "
        "same way. Find this information elsewhere: search for it, or use "
        "another site that covers the same thing (a listing, review or "
        "aggregator page)."
    ),
    "rate_limited": (
        "{host} is limiting how often it can be fetched{wait}. Use another "
        "source for now rather than retrying straight away."
    ),
    "not_found": (
        "That page does not exist; the link may be stale or mistyped. Search "
        "for the current page, or start from the site's home page."
    ),
    "rejected": (
        "{host} rejected the request, and fetching it again will fail the same "
        "way. Use another source."
    ),
    "server_error": (
        "The site had a server error, which may be temporary. One retry is "
        "reasonable; if that fails too, use another source."
    ),
    "timeout": (
        "{host} did not respond in time. One retry is reasonable; if that "
        "fails too, use another source."
    ),
    "unreachable": (
        "Could not connect to {host}. Check the URL is spelled right; the site "
        "may be down or gone, so look for another source."
    ),
    "needs_js": (
        "The page builds its content with JavaScript, which this fetcher does "
        "not run, so there is nothing to read. Look for the same information "
        "on another site."
    ),
}

_RETRYABLE: frozenset[FetchKind] = frozenset(
    {"rate_limited", "server_error", "timeout"}
)


class FetchError(WebError):
    """A page could not be read, classified so the model knows what to try next.

    The message names the failure and gives advice; ``kind`` and
    ``retryable`` let the tool loop decide whether a repeat is worth running.
    """

    def __init__(
        self,
        kind: FetchKind,
        url: str,
        *,
        status: int | None = None,
        detail: str = "",
        retry_after: float | None = None,
    ) -> None:
        self.kind: FetchKind = kind
        self.url = url
        self.status = status
        self.retry_after = retry_after
        reasons = [f"HTTP {status}"] if status is not None else []
        reasons += [detail] if detail else []
        reason = f" ({', '.join(reasons)})" if reasons else ""
        wait = f" (wait about {retry_after:g} seconds)" if retry_after else ""
        advice = _ADVICE[kind].format(host=self.host, wait=wait)
        label = kind.replace("_", " ").upper()
        super().__init__(f"could not fetch {url}: {label}{reason}.\n{advice}")

    @property
    def host(self) -> str:
        return host_of(self.url) or self.url

    @property
    def retryable(self) -> bool:
        return self.kind in _RETRYABLE


def host_of(url: str) -> str | None:
    """The host of ``url`` without ``www.``, so both spellings count as one site."""
    host = (urlparse(url.strip()).hostname or "").lower()
    return host.removeprefix("www.") or None


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


def fetch(url: str, *, max_chars: int | None = None) -> str:
    """Download ``url`` and return it as readable plain text, truncated.

    ``max_chars`` defaults per site (see :func:`page_chars`).
    """
    return truncate(read(url), max_chars or page_chars(url))


def page_chars(url: str) -> int:
    """How much of a page to show at once. Reddit threads get more room."""
    from . import reddit  # imports this module, so it cannot sit at the top

    return reddit.REDDIT_PAGE_CHARS if reddit.handles(url) else DEFAULT_PAGE_CHARS


def read(url: str) -> str:
    """Download ``url`` and return all of it as readable plain text.

    Failures raise :class:`FetchError`, which says what went wrong and what
    to try instead.
    """
    from . import reddit

    try:
        if reddit.handles(url):
            return reddit.read(url)
        response = get_guarded(url)
    except httpx.HTTPStatusError as exc:
        raise _status_error(url, exc.response) from exc
    except httpx.TimeoutException as exc:
        raise FetchError("timeout", url) from exc
    except httpx.TransportError as exc:
        detail = str(exc) or type(exc).__name__
        raise FetchError("unreachable", url, detail=detail) from exc
    content_type = response.headers.get("content-type", "")
    body = response.text
    is_html = "html" in content_type.lower()
    text = (html_to_text(body) if is_html else body).strip()
    if is_html and len(text) < _CHALLENGE_MAX_CHARS:
        if (checker := _bot_check(response)) is not None:
            raise FetchError("blocked", url, detail=f"a bot check from {checker}")
        if not text or _JS_ONLY.search(text):
            raise FetchError("needs_js", url, detail="no readable text")
    if not text:
        raise FetchError("needs_js", url, detail="no readable text")
    return text


# -- a page a part at a time ----------------------------------------------------

#: How long a page read this session is reused before it is fetched again.
PAGE_TTL = 3600.0
#: Pages a session keeps at once; the least recently used go first.
MAX_PAGES = 256


class Pages:
    """Pages read this session, kept whole so the model can page through them.

    Reading on through a long page, or searching it, then costs no request.
    """

    def __init__(self) -> None:
        self._texts: dict[str, tuple[float, str]] = {}

    def text(self, url: str) -> str:
        key = url.strip().partition("#")[0].rstrip("/")
        cached = self._texts.pop(key, None)
        if cached is not None and _now() - cached[0] < PAGE_TTL:
            self._texts[key] = cached
            return cached[1]
        text = read(url)
        self._texts[key] = (_now(), text)
        while len(self._texts) > MAX_PAGES:
            del self._texts[next(iter(self._texts))]
        return text


def window(text: str, offset: int, size: int) -> str:
    """``size`` characters of ``text`` from ``offset``, saying how to read on.

    A part ends at a line break where there is one in its second half, so
    lines are not cut in two. Raises :class:`WebError` for an offset past
    the end.
    """
    total = len(text)
    start = max(0, offset)
    if start and start >= total:
        msg = f"offset {offset} is past the end of the page ({total:,} characters)"
        raise WebError(msg)
    end = min(total, start + size)
    if end < total:
        newline = text.rfind("\n", start + size // 2, end)
        if newline != -1:
            end = newline + 1
    body = text[start:end].rstrip()
    head = f"[Continuing from character {start:,} of {total:,}.]\n\n" if start else ""
    if end >= total:
        return f"{head}{body}"
    return (
        f"{head}{body}\n\n[Characters {start:,}-{end:,} of {total:,}. To read on, "
        f"call fetch_url with offset {end}, or use find_in_page to jump to what "
        "you need.]"
    )


_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    ]
)
#: Characters shown either side of a match on a long line.
_CONTEXT = 250


def find_passages(text: str, query: str, *, limit: int = 8) -> str:
    """The lines of ``text`` that match ``query``, best first, with offsets.

    A line scores by how many of the query's words it contains, and gets a
    bonus for the whole phrase. Matching ignores case and finds words inside
    longer ones ("pasta" in "pastas"). Each passage carries an offset to read
    on from with ``fetch_url``.
    """
    phrase = " ".join(query.casefold().split())
    words = [w for w in re.findall(r"\w[\w'-]*", phrase) if w not in _STOPWORDS]
    words = list(dict.fromkeys(words or phrase.split()))
    if not words:
        msg = "find_in_page needs some words to look for"
        raise WebError(msg)
    scored: list[tuple[int, int, str]] = []
    position = 0
    for line in text.splitlines(keepends=True):
        folded = line.casefold()
        hits = [folded.find(word) for word in words if word in folded]
        if hits:
            score = len(hits) + (
                len(words) if len(words) > 1 and phrase in folded else 0
            )
            first = min(hits)
            scored.append((score, position, _passage(line, first, position)))
        position += len(line)
    if not scored:
        return (
            f"No passages match {query!r} in this page ({len(text):,} "
            "characters). Try other words (a synonym, singular or plural), or "
            "read the page with fetch_url."
        )
    best = sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    shown = "\n".join(
        passage for _, _, passage in sorted(best, key=operator.itemgetter(1))
    )
    count = len(scored)
    noun = "passage" if count == 1 else "passages"
    of = f"{len(best)} of " if len(best) < count else ""
    return (
        f"{of}{count} {noun} matching {query!r} ({len(text):,} characters in "
        f"all), as @offset: text\n{shown}\n\nRead around one with fetch_url "
        "and its offset."
    )


def _passage(line: str, first: int, position: int) -> str:
    """One matching line as ``@offset: text``, cut down around the match."""
    if len(line) <= 2 * _CONTEXT:
        return f"@{position}: {' '.join(line.split())}"
    start = max(0, first - _CONTEXT)
    end = first + _CONTEXT
    lead = "…" if start else ""
    tail = "…" if end < len(line) else ""
    return f"@{position + start}: {lead}{' '.join(line[start:end].split())}{tail}"


def _now() -> float:
    return time.monotonic()


# -- telling a refusal from a page --------------------------------------------

#: Bot checks a site may serve, by a marker in their HTML. Only short pages are
#: checked: a real page can embed a CAPTCHA widget in a form and still be full
#: of content.
_BOT_CHECKS: tuple[tuple[str, str], ...] = (
    ("challenge-platform", "Cloudflare"),
    ("cf-chl", "Cloudflare"),
    ("attention required! | cloudflare", "Cloudflare"),
    ("captcha-delivery.com", "DataDome"),
    ("px-captcha", "PerimeterX"),
    ("_incapsula_resource", "Imperva"),
    ("you don't have permission to access", "Akamai"),
    ("verify you are human", "the site"),
    ("are you a robot", "the site"),
    ("unusual traffic", "the site"),
    ("captcha", "the site"),
)
_CHALLENGE_MAX_CHARS = 2000
_JS_ONLY = re.compile(
    r"(enable|requires?|turn on) javascript|javascript (is )?(required|disabled)",
    re.IGNORECASE,
)


def _bot_check(response: httpx.Response) -> str | None:
    """Who is challenging us, if ``response`` is a bot check rather than a page."""
    if response.headers.get("cf-mitigated", "").lower() == "challenge":
        return "Cloudflare"
    html = response.text.lower()
    return next((name for marker, name in _BOT_CHECKS if marker in html), None)


def _status_error(url: str, response: httpx.Response) -> FetchError:
    status = response.status_code
    checker = _bot_check(response) if status in {403, 429, 503} else None
    if checker is not None:
        return FetchError(
            "blocked", url, status=status, detail=f"a bot check from {checker}"
        )
    if status == 429:
        wait = _retry_after(response)
        return FetchError("rate_limited", url, status=status, retry_after=wait)
    kind: FetchKind
    if status in {401, 403, 451}:
        kind = "blocked"
    elif status in {404, 410}:
        kind = "not_found"
    elif status >= 500:
        kind = "server_error"
    else:
        kind = "rejected"
    return FetchError(kind, url, status=status)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "").strip()
    return float(value) if value.isdigit() else None


def truncate(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return f"{text[:max_chars]}\n\n[truncated at {max_chars} characters]"
    return text


# -- HTTP with a guard ------------------------------------------------------


def get_guarded(
    url: str, *, headers: Mapping[str, str] = BROWSER_HEADERS
) -> httpx.Response:
    """GET ``url``, re-checking every redirect hop against the address guard."""
    current = url
    with httpx.Client(
        timeout=TIMEOUT,
        follow_redirects=False,
        headers=dict(headers),
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
