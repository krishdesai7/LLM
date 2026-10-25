from __future__ import annotations

from collections.abc import Callable
from typing import Self

import httpx
import pytest
from httpx._client import Client
from httpx._models import Request, Response

from llm import reddit, web

# Trimmed from a real Redlib thread page: site chrome, a post, and a comment
# with a nested reply.
THREAD_HTML = """
<nav><a href="/">redlib.</a> Feeds MAIN FEEDS</nav>
<div class="post highlighted">
  <p class="post_header">
    <a class="post_subreddit" href="/r/FoodNYC">r/FoodNYC</a>
    <span class="dot">&bull;</span>
    <a class="post_author " href="/user/CJ9103">u/CJ9103</a>
    <span class="created" title="May 11 2025, 12:29:29 UTC">May 11 &#39;25</span>
  </p>
  <h1 class="post_title">
    <a href="/r/FoodNYC/search?q=flair" class="post_flair"><span>Question</span></a>
    Dinner recommendations (Italian, Burger, Steak)
  </h1>
  <div class="post_body"><div class="md"><p>Visiting in June.</p>
<ul>
<li><strong>Cuisine:</strong> Italian</li>
<li>Rubirosa</li>
</ul>
</div></div>
  <div class="post_score" title="5">5<span class="label"> Upvotes</span></div>
  <div class="post_footer"><ul id="post_links"><li>
    <a href="#popup">reddit</a>
    <div class="popup" id="popup"><div class="popup-inner">
      <h1>You are about to leave Redlib</h1><p>Do you want to continue?</p>
      <svg><style>.cls-1{fill:#000}</style></svg>
    </div></div>
  </li></ul></div>
</div>
<div class="thread">
<div id="a1" class="comment">
  <div class="comment_left">
    <p class="comment_score" title="2">2</p><div class="line"></div>
  </div>
  <details class="comment_right" open>
    <summary class="comment_data">
      <a class="comment_author  " href="/user/Garcon">u/Garcon</a>
      <a href="/r/FoodNYC/comments/1kjzwrv/x/a1/" class="created"
         title="May 11 2025, 15:03:48 UTC">May 11 &#39;25</a>
    </summary>
    <div class="comment_body "><div class="md"><p>Authentic Italian:</p>
<p>Via Carota<br>Rezdôra</p>
</div></div>
    <blockquote class="replies">
<div id="a2" class="comment">
  <div class="comment_left"><p class="comment_score" title="1">1</p></div>
  <details class="comment_right" open>
    <summary class="comment_data">
      <a class="comment_author  op" href="/user/CJ9103">u/CJ9103</a>
      <a href="#" class="created" title="May 11 2025, 18:48:37 UTC">May 11 &#39;25</a>
    </summary>
    <div class="comment_body "><div class="md"><p>Thank you!!</p></div></div>
  </details>
</div>
    </blockquote>
  </details>
</div>
</div>
"""

LISTING_HTML = """
<div class="post" id="1gxr0k4">
  <p class="post_header">
    <a class="post_subreddit" href="/r/FoodNYC">r/FoodNYC</a>
    <a class="post_author " href="/u/sohoships">u/sohoships</a>
    <span class="created" title="Nov 23 2024, 04:05:58 UTC">Nov 23 &#39;24</span>
  </p>
  <h2 class="post_title">
    <a href="/r/FoodNYC/comments/1gxr0k4/why_is_italian/"
      >Why is Italian food so overpriced?</a>
  </h2>
  <div class="post_score" title="968">968<span class="label"> Upvotes</span></div>
  <div class="post_body post_preview"><div class="md"><p>My take on it.</p></div></div>
  <div class="post_footer">
    <a href="/r/FoodNYC/comments/1gxr0k4/why_is_italian/" class="post_comments"
       title="417 comments">417 comments</a>
  </div>
</div>
"""

FEED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Dinner recommendations (Italian, Burger, Steak) : FoodNYC</title>
<entry>
  <author><name>/u/CJ9103</name></author>
  <content type="html">&lt;div class=&quot;md&quot;&gt;
    &lt;p&gt;Visiting in June.&lt;/p&gt;&lt;/div&gt;</content>
  <link href="https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/x/"/>
  <updated>2025-05-11T12:29:29+00:00</updated>
  <title>Dinner recommendations (Italian, Burger, Steak)</title>
</entry>
<entry>
  <author><name>/u/Garcon</name></author>
  <content type="html">&lt;p&gt;Via Carota&lt;/p&gt;</content>
  <link href="https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/x/a1/"/>
  <updated>2025-05-11T15:03:48+00:00</updated>
  <title>/u/Garcon on Dinner recommendations (Italian, Burger, Steak)</title>
</entry>
</feed>
"""


@pytest.fixture(autouse=True)
def _no_instance(monkeypatch) -> None:
    monkeypatch.delenv(reddit.REDDIT_URL_VAR, raising=False)
    monkeypatch.setattr(reddit.time, "sleep", lambda _seconds: None)
    # no test may start a real Redlib
    monkeypatch.setattr(reddit.shutil, "which", lambda _name: None)
    monkeypatch.setattr(reddit, "_launched", [])


class _FakeRedlib:
    """Stands in for ``subprocess.Popen``; Redlib "answers" once launched."""

    def __init__(self, monkeypatch, *, exits: bool = False) -> None:
        self.launches: list[tuple[list[str], dict]] = []
        self.exits = exits
        self.stopped_at_exit: list[object] = []
        monkeypatch.setattr(reddit.shutil, "which", lambda _name: "/bin/redlib")
        monkeypatch.setattr(reddit.subprocess, "Popen", self._popen)
        monkeypatch.setattr(reddit.atexit, "register", self._at_exit)
        monkeypatch.setattr(reddit, "_answering", lambda *_a, **_k: self.running)

    @property
    def running(self) -> bool:
        return bool(self.launches) and not self.exits

    def _popen(self, command, **kwargs) -> Self:
        self.launches.append((command, kwargs))
        return self

    def _at_exit(self, _func, proc) -> None:
        self.stopped_at_exit.append(proc)

    def poll(self) -> int | None:
        return 1 if self.exits else None


def _thread_once_running(redlib: _FakeRedlib) -> Callable[[Request], Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.reddit.com":
            return httpx.Response(200, text=FEED_XML)
        if not redlib.running:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, html=THREAD_HTML)

    return handler


def _serve(monkeypatch, handler) -> list[httpx.Request]:
    """Answer every HTTP request in-process, recording what was asked."""
    seen: list[httpx.Request] = []
    monkeypatch.setattr(web, "_assert_public", lambda _url: None)
    real_client = httpx.Client

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    def client(**kwargs) -> Client:
        kwargs["transport"] = httpx.MockTransport(recording)
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    return seen


# -- which URLs are ours ----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/",
        "https://old.reddit.com/r/FoodNYC/",
        "https://reddit.com/r/FoodNYC/",
        "https://redd.it/1kjzwrv",
    ],
)
def test_reddit_urls_are_handled(url: str) -> None:
    assert reddit.handles(url)


@pytest.mark.parametrize(
    "url", ["https://example.com/r/x", "https://notreddit.com/", "https://reddit.co/"]
)
def test_other_urls_are_not(url: str) -> None:
    assert not reddit.handles(url)


# -- rendering Redlib pages -------------------------------------------------


def test_thread_renders_post_and_comment_tree() -> None:
    text = reddit.render_redlib(THREAD_HTML)
    assert text.splitlines()[:2] == [
        "r/FoodNYC · u/CJ9103 · May 11 2025 · 5 points",
        "# Dinner recommendations (Italian, Burger, Steak)",
    ]
    assert "- Cuisine: Italian\n- Rubirosa" in text
    assert (
        "[2] u/Garcon · May 11 2025: Authentic Italian:\n  Via Carota\n  Rezdôra"
        in text
    )
    assert "\n  [1] u/CJ9103 (OP) · May 11 2025: Thank you!!" in text


def test_site_chrome_is_dropped() -> None:
    text = reddit.render_redlib(THREAD_HTML)
    for noise in ("redlib.", "MAIN FEEDS", "leave Redlib", "cls-1", "Question"):
        assert noise not in text


def test_listing_renders_each_post_with_a_reddit_link() -> None:
    text = reddit.render_redlib(LISTING_HTML)
    assert text.splitlines() == [
        "[968] Why is Italian food so overpriced?",
        "    r/FoodNYC · u/sohoships · Nov 23 2024 · 417 comments",
        "    https://www.reddit.com/r/FoodNYC/comments/1gxr0k4/why_is_italian/",
        "    My take on it.",
    ]


def test_unrecognised_pages_render_nothing() -> None:
    assert reddit.render_redlib("<p>Something else entirely</p>") == ""


# -- the Redlib route -------------------------------------------------------


def test_fetch_goes_through_the_configured_instance(monkeypatch) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, "http://127.0.0.1:8089/")
    seen = _serve(monkeypatch, lambda _r: httpx.Response(200, html=THREAD_HTML))
    text = web.fetch("https://old.reddit.com/r/FoodNYC/comments/1kjzwrv/?sort=top")
    assert (
        str(seen[0].url) == "http://127.0.0.1:8089/r/FoodNYC/comments/1kjzwrv/?sort=top"
    )
    assert "u/Garcon" in text


def test_short_links_map_to_the_comments_route(monkeypatch) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, "http://127.0.0.1:8089")
    seen = _serve(monkeypatch, lambda _r: httpx.Response(200, html=THREAD_HTML))
    web.fetch("https://redd.it/1kjzwrv")
    assert seen[0].url.path == "/comments/1kjzwrv"


def test_instance_errors_are_reported(monkeypatch) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, "http://127.0.0.1:8089")
    _serve(monkeypatch, lambda _r: httpx.Response(404, html="<p>Post not found</p>"))
    with pytest.raises(web.WebError, match="404"):
        web.fetch("https://www.reddit.com/r/x/comments/zzz/")


def test_an_unreachable_instance_falls_back_to_the_feed(monkeypatch) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, "http://127.0.0.1:8089")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "127.0.0.1":
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, text=FEED_XML)

    _serve(monkeypatch, handler)
    text = web.fetch("https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/")
    assert "unreachable" in text
    assert "Via Carota" in text


# -- starting Redlib on demand ----------------------------------------------

THREAD_URL = "https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/"


def test_a_local_instance_that_is_down_gets_started(monkeypatch) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, "http://localhost:8089")
    redlib = _FakeRedlib(monkeypatch)
    _serve(monkeypatch, _thread_once_running(redlib))
    text = web.fetch(THREAD_URL)
    assert "u/Garcon" in text
    assert "unreachable" not in text
    [(command, options)] = redlib.launches
    assert command == ["/bin/redlib", "--address", "127.0.0.1", "--port", "8089"]
    assert options["start_new_session"]
    assert redlib.stopped_at_exit == [redlib]


def test_it_is_started_at_most_once_per_session(monkeypatch) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, "http://127.0.0.1:8089")
    redlib = _FakeRedlib(monkeypatch)
    _serve(monkeypatch, _thread_once_running(redlib))
    web.fetch(THREAD_URL)
    web.fetch(THREAD_URL)
    assert len(redlib.launches) == 1


def test_a_redlib_that_exits_falls_back_to_the_feed(monkeypatch) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, "http://127.0.0.1:8089")
    redlib = _FakeRedlib(monkeypatch, exits=True)
    _serve(monkeypatch, _thread_once_running(redlib))
    text = web.fetch(THREAD_URL)
    assert "unreachable" in text
    assert "Via Carota" in text


@pytest.mark.parametrize(
    "base", ["http://redlib.example.com:8089", "http://127.0.0.1", "http://10.0.0.5:80"]
)
def test_only_a_local_url_with_a_port_is_started(monkeypatch, base: str) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, base)
    redlib = _FakeRedlib(monkeypatch)
    _serve(monkeypatch, _thread_once_running(redlib))
    assert "unreachable" in web.fetch(THREAD_URL)
    assert redlib.launches == []


def test_status_says_when_it_will_be_started(monkeypatch) -> None:
    monkeypatch.setenv(reddit.REDDIT_URL_VAR, "http://127.0.0.1:8089")
    _FakeRedlib(monkeypatch)
    assert "starts on the first link" in reddit.status()


# -- the feed route ---------------------------------------------------------


def test_without_an_instance_the_feed_is_used(monkeypatch) -> None:
    seen = _serve(monkeypatch, lambda _r: httpx.Response(200, text=FEED_XML))
    text = web.fetch("https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/")
    assert str(seen[0].url) == "https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/.rss"
    assert text.splitlines()[0] == "# Dinner recommendations (Italian, Burger, Steak)"
    assert "u/CJ9103 · 2025-05-11: Visiting in June." in text
    assert "u/Garcon · 2025-05-11: Via Carota" in text


def test_feed_urls_keep_the_query(monkeypatch) -> None:
    seen = _serve(monkeypatch, lambda _r: httpx.Response(200, text=FEED_XML))
    web.fetch("https://www.reddit.com/r/FoodNYC/search?q=italian&restrict_sr=on")
    assert str(seen[0].url) == (
        "https://www.reddit.com/r/FoodNYC/search.rss?q=italian&restrict_sr=on"
    )


def test_the_feed_rate_limit_is_waited_out_once(monkeypatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr(reddit.time, "sleep", waits.append)
    replies = iter(
        [
            httpx.Response(429, headers={"x-ratelimit-reset": "42"}),
            httpx.Response(200, text=FEED_XML),
        ]
    )
    _serve(monkeypatch, lambda _r: next(replies))
    assert "Via Carota" in web.fetch(
        "https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/"
    )
    assert waits == [42.0]


def test_a_persistent_rate_limit_is_explained(monkeypatch) -> None:
    _serve(
        monkeypatch, lambda _r: httpx.Response(429, headers={"x-ratelimit-reset": "9"})
    )
    with pytest.raises(web.WebError, match=reddit.REDDIT_URL_VAR):
        web.fetch("https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/")


def test_a_non_feed_reply_is_an_error(monkeypatch) -> None:
    _serve(monkeypatch, lambda _r: httpx.Response(200, html="<html>login</html>"))
    with pytest.raises(web.WebError, match="feed"):
        web.fetch("https://www.reddit.com/r/FoodNYC/comments/1kjzwrv/")
