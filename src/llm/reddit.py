"""Reading Reddit, which turns away ordinary fetches.

Scripted requests to reddit.com get a JavaScript challenge, a login wall or a
403, so :func:`llm.web.fetch` hands Reddit URLs here. Two routes:

* a self-hosted Redlib instance (``$LLM_REDDIT_URL``) serves the whole thread,
  and its markup is parsed into a compact comment tree. If that URL is on this
  machine and nothing is answering, the ``redlib`` binary is started for the
  rest of the session;
* otherwise Reddit's Atom feeds, which still answer anonymously but allow
  about one request a minute and carry only some of the comments.
"""

from __future__ import annotations

import atexit
import os
import shutil

# ruff: ignore[suspicious-subprocess-import] - only to start a local Redlib
import subprocess
import time

# ruff: ignore[suspicious-xml-etree-import] - ElementTree never resolves external
# entities and expat (>= 2.4.1) caps entity expansion; input is Reddit's feed
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

from . import web

__all__ = ["REDDIT_URL_VAR", "fetch", "handles", "instance_url", "render_redlib"]

REDDIT_URL_VAR = "LLM_REDDIT_URL"
#: Threads are dense with useful text, so they get more room than other pages.
REDDIT_PAGE_CHARS = 12_000
#: Longest we will sit out the feed's rate limit before giving up.
FEED_MAX_WAIT = 65.0
#: How long a Redlib we started gets to begin answering (it takes about 2s).
REDLIB_START_WAIT = 10.0

_SITE = "https://www.reddit.com"
#: Loopback host names, and the address Redlib should bind for each.
_LOCAL_HOSTS = {"127.0.0.1": "127.0.0.1", "localhost": "127.0.0.1", "::1": "::1"}
#: The Redlib this session started, if any; stopped when the session exits.
_launched: list[subprocess.Popen[bytes]] = []


def handles(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"reddit.com", "redd.it"} or host.endswith(".reddit.com")


def instance_url() -> str | None:
    return os.environ.get(REDDIT_URL_VAR, "").strip().rstrip("/") or None


def status() -> str:
    """One line for ``llm web``: which route Reddit links take."""
    base = instance_url()
    if base is None:
        return f"reddit via its feeds (~1 page/min); set {REDDIT_URL_VAR} for Redlib"
    if _answering(base, timeout=2.0):
        return f"reddit via Redlib at {base}"
    if _launch_command(base) is not None:
        return f"reddit via Redlib at {base} (not running; starts on the first link)"
    return f"reddit via Redlib at {base}, which is not responding (feeds as fallback)"


def fetch(url: str, *, max_chars: int = REDDIT_PAGE_CHARS) -> str:
    """A Reddit page as readable text, by whichever route is available."""
    path = _path_of(url)
    notice = ""
    base = instance_url()
    if base is not None:
        text = _try_redlib(base, path)
        if text is None and _start_redlib(base):
            text = _try_redlib(base, path)
        if text is not None:
            return web.truncate(text, max_chars)
        notice = (
            f"(Redlib at {base} is unreachable, so this came from Reddit's "
            "feed, which carries only some of the comments.)\n\n"
        )
    return notice + web.truncate(_via_feed(path), max_chars)


def _path_of(url: str) -> str:
    """The path and query to request, normalising ``redd.it`` short links."""
    parsed = urlparse(url)
    path = parsed.path or "/"
    if (parsed.hostname or "").lower() == "redd.it":
        path = f"/comments{path}"
    return f"{path}?{parsed.query}" if parsed.query else path


# -- Redlib -----------------------------------------------------------------

# The instance is configured by the user, not chosen by the model, so like the
# search backends it skips the public-address guard.


def _try_redlib(base: str, path: str) -> str | None:
    """The page from Redlib, or None if nothing is listening at ``base``."""
    try:
        return _via_redlib(base, path)
    except httpx.TransportError:
        return None


def _answering(base: str, *, timeout: float) -> bool:
    try:
        httpx.get(base, timeout=timeout)
    except httpx.TransportError:
        return False
    return True


def _launch_command(base: str) -> list[str] | None:
    """How to start Redlib at ``base``, if it is on this machine and installed."""
    parsed = urlparse(base)
    address = _LOCAL_HOSTS.get(parsed.hostname or "")
    binary = shutil.which("redlib")
    if address is None or parsed.port is None or binary is None:
        return None
    return [binary, "--address", address, "--port", str(parsed.port)]


def _start_redlib(base: str) -> bool:
    """Start Redlib for the rest of the session; True once it is answering.

    This also covers a Redlib that another chat started and took with it on
    exit. One we started ourselves that has stopped answering is left alone.
    """
    command = _launch_command(base)
    if command is None or any(proc.poll() is None for proc in _launched):
        return False
    # ruff: ignore[subprocess-without-shell-equals-true] - fixed argv; the URL
    # comes from the user's environment and only its loopback port is used
    proc = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # its own process group, so ctrl-c in the chat doesn't reach it
        start_new_session=True,
    )
    _launched.append(proc)
    atexit.register(_stop, proc)
    deadline = time.monotonic() + REDLIB_START_WAIT
    while proc.poll() is None and time.monotonic() < deadline:
        if _answering(base, timeout=1.0):
            return True
        time.sleep(0.1)
    return False


def _stop(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _via_redlib(base: str, path: str) -> str:
    with httpx.Client(
        timeout=web.TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": web.USER_AGENT},
    ) as client:
        response = client.get(f"{base}{path}")
    if response.is_error:
        msg = f"Redlib could not load {path} (HTTP {response.status_code})"
        raise web.WebError(msg)
    return render_redlib(response.text) or web.html_to_text(response.text)


@dataclass(slots=True)
class _Post:
    title: str = ""
    url: str = ""
    subreddit: str = ""
    author: str = ""
    created: str = ""
    score: str = ""
    comments: str = ""
    body: str = ""
    #: Redlib marks the post a thread page is about.
    highlighted: bool = False


@dataclass(slots=True)
class _Comment:
    depth: int
    author: str = ""
    created: str = ""
    score: str = ""
    body: str = ""
    op: bool = False


@dataclass(slots=True)
class _Open:
    """An element still open in the parse, and what its text is for."""

    tag: str
    field: str | None = None
    owner: _Post | _Comment | None = None


_VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "source", "track", "wbr",
})  # fmt: skip
_SKIP_TAGS = frozenset({"script", "style", "svg", "noscript", "template", "form"})
#: Redlib class -> the field its text fills.
_TEXT_FIELDS = {
    "post_title": "title",
    "post_subreddit": "subreddit",
    "post_author": "author",
    "comment_author": "author",
    "post_body": "body",
    "comment_body": "body",
}
#: Redlib class -> the field filled from the element's ``title`` attribute.
_ATTR_FIELDS = {
    "created": "created",
    "post_score": "score",
    "comment_score": "score",
    "post_comments": "comments",
}
_BLOCK_TAGS = frozenset({"p", "div", "blockquote", "pre", "h1", "h2", "h3", "tr"})


class _RedlibParser(HTMLParser):
    """Pull posts and comments out of a Redlib page, ignoring everything else."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.posts: list[_Post] = []
        self.comments: list[_Comment] = []
        self._open: list[_Open] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        owner = self._owner()
        if tag == "div" and "post" in classes:
            post = _Post(highlighted="highlighted" in classes)
            self.posts.append(post)
            self._push(tag, owner=post)
            return
        if tag == "div" and "comment" in classes:
            depth = sum(isinstance(entry.owner, _Comment) for entry in self._open)
            comment = _Comment(depth)
            self.comments.append(comment)
            self._push(tag, owner=comment)
            return
        if owner is None:
            self._push(tag)
            return
        if tag in _SKIP_TAGS or classes & {"popup", "post_flair"}:
            self._push(tag, field="skip")
            return
        for name, attr_field in _ATTR_FIELDS.items():
            if name in classes and not getattr(owner, attr_field):
                setattr(owner, attr_field, _short_date(values.get("title") or ""))
        if isinstance(owner, _Comment) and "op" in classes:
            owner.op = True
        if isinstance(owner, _Post) and tag == "a" and self._field() == "title":
            owner.url = _SITE + (values.get("href") or "")
        if self._field() == "body":
            self._break(owner, tag)
        text_field = next((f for c, f in _TEXT_FIELDS.items() if c in classes), None)
        self._push(tag, field=text_field)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if self._field() == "body" and tag in _BLOCK_TAGS | {"li"}:
            owner = self._owner()
            if owner is not None:
                owner.body += "\n"
        for index in range(len(self._open) - 1, -1, -1):
            if self._open[index].tag == tag:
                del self._open[index:]
                return

    def handle_data(self, data: str) -> None:
        name, owner = self._field(), self._owner()
        if name in {None, "skip"} or owner is None:
            return
        current = getattr(owner, name)
        setattr(owner, name, current + data)

    def _push(
        self,
        tag: str,
        field: str | None = None,
        owner: _Post | _Comment | None = None,
    ) -> None:
        if tag not in _VOID_TAGS:
            self._open.append(_Open(tag, field, owner))

    def _field(self) -> str | None:
        """The innermost text field we are inside, if any."""
        for entry in reversed(self._open):
            if entry.field is not None:
                return entry.field
            if entry.owner is not None:
                return None
        return None

    def _owner(self) -> _Post | _Comment | None:
        for entry in reversed(self._open):
            if entry.owner is not None:
                return entry.owner
        return None

    @staticmethod
    def _break(owner: _Post | _Comment, tag: str) -> None:
        """Keep paragraph and list structure inside a body."""
        if tag == "li":
            owner.body += "\n- "
        elif tag == "br" or tag in _BLOCK_TAGS:
            owner.body += "\n"


def render_redlib(html: str) -> str:
    """A Redlib thread or listing as compact text, or ``""`` if it is neither."""
    parser = _RedlibParser()
    parser.feed(html)
    posts, comments = parser.posts, parser.comments
    if not posts:
        return ""
    thread = next((post for post in posts if post.highlighted), None)
    if thread is not None:
        return _render_thread(thread, comments)
    return "\n\n".join(_render_listed(post) for post in posts)


def _render_thread(post: _Post, comments: list[_Comment]) -> str:
    meta = [post.subreddit, post.author, post.created]
    if post.score:
        meta.append(f"{post.score} points")
    lines = [_joined(meta), f"# {_flat(post.title)}"]
    if body := _block(post.body):
        lines += ["", body]
    if comments:
        lines += ["", "Comments:"]
    for comment in comments:
        indent = "  " * comment.depth
        author = _flat(comment.author) + (" (OP)" if comment.op else "")
        head = f"{indent}[{comment.score}] {_joined([author, comment.created])}: "
        lines.append(head + _block(comment.body).replace("\n", f"\n{indent}  "))
    return "\n".join(lines)


def _render_listed(post: _Post) -> str:
    lines = [
        f"[{post.score}] {_flat(post.title)}",
        f"    {_joined([post.subreddit, post.author, post.created, post.comments])}",
    ]
    if post.url:
        lines.append(f"    {post.url}")
    if preview := _flat(post.body):
        lines.append(f"    {preview[:300]}")
    return "\n".join(lines)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _block(text: str) -> str:
    """Collapse runs of whitespace but keep line breaks."""
    lines = (_flat(line) for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _joined(parts: list[str]) -> str:
    return " · ".join(_flat(part) for part in parts if _flat(part))


def _short_date(title: str) -> str:
    """``May 11 2025, 12:29:29 UTC`` -> ``May 11 2025``; other values pass."""
    return title.split(",", 1)[0].strip()


# -- Atom feeds -------------------------------------------------------------

_ATOM = "{http://www.w3.org/2005/Atom}"


def _via_feed(path: str) -> str:
    route, _, query = path.partition("?")
    url = f"{_SITE}{route}.rss" + (f"?{query}" if query else "")
    for attempt in range(2):
        try:
            response = web.get_guarded(url)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 429:
                raise
            if attempt == 0:
                time.sleep(_reset_seconds(exc.response))
            continue
        return render_feed(response.text, thread="/comments/" in route)
    msg = (
        "Reddit's feed is rate-limited (about one page a minute). Set "
        f"{REDDIT_URL_VAR} to a Redlib instance to read Reddit without the limit."
    )
    raise web.WebError(msg)


def _reset_seconds(response: httpx.Response) -> float:
    try:
        reset = float(response.headers.get("x-ratelimit-reset", ""))
    except ValueError:
        reset = 60.0
    return min(max(reset, 1.0), FEED_MAX_WAIT)


def render_feed(xml: str, *, thread: bool) -> str:
    """A Reddit Atom feed as text: a post and its comments, or a listing."""
    try:
        # ruff: ignore[suspicious-xml-element-tree-usage] - expat refuses
        # external entities, and this is Reddit's own feed
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        msg = f"Reddit did not return a feed ({exc})"
        raise web.WebError(msg) from exc
    if root.tag != f"{_ATOM}feed":
        msg = "Reddit did not return a feed (probably a login or block page)"
        raise web.WebError(msg)
    entries = [_FeedEntry.of(entry) for entry in root.iter(f"{_ATOM}entry")]
    if thread and entries:
        post, *comments = entries
        lines = [f"# {post.title}", "", post.line()]
        if comments:
            lines += ["", "Comments (only some are in the feed):"]
        lines += [comment.line() for comment in comments]
        return "\n".join(lines)
    return "\n\n".join(
        f"{entry.title}\n    {entry.link}\n    {_flat(entry.body)[:300]}"
        for entry in entries
    )


@dataclass(frozen=True, slots=True)
class _FeedEntry:
    author: str
    updated: str
    title: str
    body: str
    link: str

    @classmethod
    def of(cls, entry: ET.Element) -> _FeedEntry:
        link = entry.find(f"{_ATOM}link")
        return cls(
            author=(entry.findtext(f"{_ATOM}author/{_ATOM}name") or "").lstrip("/"),
            updated=(entry.findtext(f"{_ATOM}updated") or "")[:10],
            title=_flat(entry.findtext(f"{_ATOM}title") or ""),
            body=web.html_to_text(entry.findtext(f"{_ATOM}content") or ""),
            link=link.get("href", "") if link is not None else "",
        )

    def line(self) -> str:
        return f"{_joined([self.author, self.updated])}: " + self.body.replace(
            "\n", "\n  "
        )
