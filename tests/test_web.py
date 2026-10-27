from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from llm import web

if TYPE_CHECKING:
    from httpx._client import Client

HTML = """
<html><head><title>t</title><style>.a{color:red}</style></head>
<body>
  <script>var x = 1;</script>
  <h1>Heading</h1>
  <p>First   paragraph.</p>
  <p>Second <b>paragraph</b> with a <a href="/x">link</a>.</p>
</body></html>
"""


@pytest.fixture(autouse=True)
def _no_backend_env(monkeypatch) -> None:
    for backend in web.backends():
        for var in (backend.env_var, backend.url_var):
            if var:
                monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("LLM_SEARCH_BACKEND", raising=False)


# -- html extraction --------------------------------------------------------


def test_scripts_and_styles_are_dropped() -> None:
    text = web.html_to_text(HTML)
    assert "var x" not in text
    assert "color:red" not in text


def test_readable_text_survives() -> None:
    text = web.html_to_text(HTML)
    assert "Heading" in text
    assert "First paragraph." in text
    assert "Second paragraph with a link." in text.replace("\n", " ")


def test_whitespace_is_collapsed() -> None:
    assert "\n\n\n" not in web.html_to_text(HTML)


# -- the address guard ------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/admin",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://[::1]/",
    ],
)
def test_private_addresses_are_refused(url: str) -> None:
    with pytest.raises(web.WebError, match="not a public address"):
        web.fetch(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x"])
def test_non_http_schemes_are_refused(url: str) -> None:
    with pytest.raises(web.WebError, match="only http"):
        web.fetch(url)


def test_unresolvable_hosts_are_reported(monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise OSError("nodename nor servname provided")

    monkeypatch.setattr(web.socket, "getaddrinfo", boom)
    with pytest.raises(web.WebError, match="could not resolve"):
        web.fetch("https://nope.invalid/")


# -- fetching ---------------------------------------------------------------


def _serve(monkeypatch, handler) -> None:
    """Point the fetcher at an in-process transport, bypassing DNS."""
    monkeypatch.setattr(web, "_assert_public", lambda _url: None)
    real_client = httpx.Client

    def client(**kwargs) -> Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(web.httpx, "Client", client)


def test_fetch_returns_extracted_text(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, html=HTML, headers={"content-type": "text/html; charset=utf-8"}
        )

    _serve(monkeypatch, handler)
    text = web.fetch("https://example.com/")
    assert "Heading" in text
    assert "<h1>" not in text


def test_pages_are_requested_the_way_a_browser_would(monkeypatch) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text="ok", headers={"content-type": "text/pl"})

    _serve(monkeypatch, handler)
    web.fetch("https://example.com/")
    assert seen[0].headers["user-agent"].startswith("Mozilla/5.0")
    assert "Chrome/" in seen[0].headers["user-agent"]
    assert "text/html" in seen[0].headers["accept"]
    assert seen[0].headers["accept-language"].startswith("en")


def test_fetch_truncates_long_pages(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 5000, headers={"content-type": "text/pl"})

    _serve(monkeypatch, handler)
    text = web.fetch("https://example.com/", max_chars=100)
    assert text.startswith("x" * 100)
    assert "[truncated at 100 characters]" in text


def test_fetch_follows_redirects_up_to_a_limit(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/final":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(200, text="arrived", headers={"content-type": "text/pl"})

    _serve(monkeypatch, handler)
    assert web.fetch("https://example.com/start") == "arrived"


def test_endless_redirects_give_up(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/again"})

    _serve(monkeypatch, handler)
    with pytest.raises(web.WebError, match="too many redirects"):
        web.fetch("https://example.com/")


def test_oversized_responses_are_refused(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="x",
            headers={"content-type": "text/pl", "content-length": "99999999"},
        )

    _serve(monkeypatch, handler)
    with pytest.raises(web.WebError, match="over the"):
        web.fetch("https://example.com/")


# -- failures the model can act on -------------------------------------------

CLOUDFLARE_CHALLENGE = """
<html><head><title>Just a moment...</title></head><body>
<div id="challenge-running">Checking your browser before accessing.</div>
<script src="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1"></script>
</body></html>
"""


def _failure(monkeypatch, handler, url: str = "https://www.example.com/menu"):
    _serve(monkeypatch, handler)
    with pytest.raises(web.FetchError) as info:
        web.fetch(url)
    return info.value


def _status(code: int, text: str = "nope", **headers: str):
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, text=text, headers=headers)

    return handler


def test_a_403_is_a_block_that_says_not_to_retry(monkeypatch) -> None:
    error = _failure(monkeypatch, _status(403))
    assert error.kind == "blocked"
    assert error.status == 403
    assert not error.retryable
    message = str(error)
    assert "HTTP 403" in message
    assert "example.com" in message
    assert "fail the same way" in message


def test_a_cloudflare_challenge_served_as_200_is_still_a_block(monkeypatch) -> None:
    error = _failure(
        monkeypatch,
        _status(200, CLOUDFLARE_CHALLENGE, **{"content-type": "text/html"}),
    )
    assert error.kind == "blocked"
    assert "Cloudflare" in str(error)


def test_a_challenge_header_marks_a_503_as_a_block(monkeypatch) -> None:
    error = _failure(monkeypatch, _status(503, **{"cf-mitigated": "challenge"}))
    assert error.kind == "blocked"


def test_a_long_page_that_mentions_captcha_is_not_a_block(monkeypatch) -> None:
    body = "<html><body><p>" + "Real menu item. " * 300 + "</p>"
    body += '<form><div class="g-recaptcha"></div></form></body></html>'
    _serve(monkeypatch, _status(200, body, **{"content-type": "text/html"}))
    assert "Real menu item." in web.fetch("https://example.com/")


def test_a_429_is_rate_limiting_with_the_wait(monkeypatch) -> None:
    error = _failure(monkeypatch, _status(429, **{"retry-after": "30"}))
    assert error.kind == "rate_limited"
    assert error.retry_after == pytest.approx(30)
    assert "30 seconds" in str(error)


@pytest.mark.parametrize("code", [404, 410])
def test_missing_pages_are_not_found(monkeypatch, code: int) -> None:
    error = _failure(monkeypatch, _status(code))
    assert error.kind == "not_found"
    assert not error.retryable


def test_server_errors_are_worth_one_retry(monkeypatch) -> None:
    error = _failure(monkeypatch, _status(502))
    assert error.kind == "server_error"
    assert error.retryable


def test_timeouts_are_classified(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    error = _failure(monkeypatch, handler)
    assert error.kind == "timeout"
    assert error.retryable


def test_refused_connections_are_unreachable(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    error = _failure(monkeypatch, handler)
    assert error.kind == "unreachable"
    assert "connection refused" in str(error)


def test_a_page_with_no_text_needs_javascript(monkeypatch) -> None:
    body = "<html><body><div id='root'></div><script>app()</script></body></html>"
    error = _failure(monkeypatch, _status(200, body, **{"content-type": "text/html"}))
    assert error.kind == "needs_js"


def test_the_host_ignores_www() -> None:
    error = web.FetchError("blocked", "https://www.example.com/a", status=403)
    assert error.host == "example.com"


# -- backend selection ------------------------------------------------------


def test_the_keyless_backend_is_the_fallback() -> None:
    assert web.active_backend().name == "duckduckgo"


def test_a_configured_api_wins(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    assert web.active_backend().name == "tavily"


def test_priority_order_is_respected(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-x")
    assert web.active_backend().name == "brave"


def test_the_backend_can_be_pinned(monkeypatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-x")
    monkeypatch.setenv("LLM_SEARCH_BACKEND", "exa")
    assert web.active_backend().name == "exa"


def test_an_unknown_pin_is_an_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_SEARCH_BACKEND", "altavista")
    with pytest.raises(web.WebError, match="unknown LLM_SEARCH_BACKEND"):
        web.active_backend()


def test_a_pinned_but_unconfigured_backend_explains_itself(monkeypatch) -> None:
    monkeypatch.setenv("LLM_SEARCH_BACKEND", "brave")
    with pytest.raises(web.WebError, match="not configured"):
        web.search("anything")


def test_searxng_needs_an_instance_url(monkeypatch) -> None:
    backend = next(b for b in web.backends() if b.name == "searxng")
    assert not backend.is_ready()
    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8888")
    assert backend.is_ready()


# -- backend requests -------------------------------------------------------


def test_brave_results_are_normalised(monkeypatch) -> None:
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-x")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["token"] = request.headers.get("x-subscription-token")
        seen["query"] = request.url.params.get("q")
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": "Python 3.14",
                            "url": "https://example.com/py",
                            "description": "What's <b>new</b>",
                        }
                    ]
                }
            },
        )

    _serve(monkeypatch, handler)
    results = web.search("python 3.14")
    assert results == [
        web.Result("Python 3.14", "https://example.com/py", "What's new")
    ]
    assert seen == {"token": "brv-x", "query": "python 3.14"}


def test_search_failures_become_web_errors(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    _serve(monkeypatch, handler)
    with pytest.raises(web.WebError, match="request failed"):
        web.search("anything")


def test_results_are_capped_at_the_requested_count(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": f"r{i}", "url": f"https://e/{i}", "content": ""}
                    for i in range(10)
                ]
            },
        )

    _serve(monkeypatch, handler)
    assert len(web.search("q", count=3)) == 3


def test_duckduckgo_html_is_parsed(monkeypatch) -> None:
    page = """
    <table>
      <tr><td><a rel="nofollow" class="result-link" href="https://a.example/">A
        title</a></td></tr>
      <tr><td class="result-snippet">A snippet.</td></tr>
      <tr><td><a class="result-link" href="https://b.example/">B</a></td></tr>
      <tr><td class="result-snippet">B snippet.</td></tr>
    </table>
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=page)

    _serve(monkeypatch, handler)
    results = web.search("q")
    assert [r.url for r in results] == ["https://a.example/", "https://b.example/"]
    assert results[0].title == "A title"
    assert results[0].snippet == "A snippet."


def test_rendered_results_are_readable() -> None:
    rendered = web.render([web.Result("T", "https://e/", "snip")])
    assert rendered == "[1] T\n    https://e/\n    snip"


def test_rendering_nothing_says_so() -> None:
    assert web.render([]) == "No results."


# -- reading a page a part at a time ------------------------------------------------


def _lines(count: int, width: int = 99) -> str:
    """``count`` numbered lines, each ``width`` characters plus a newline."""
    return "".join(f"{i:04d}" + "x" * (width - 4) + "\n" for i in range(count)).rstrip()


def test_read_returns_the_whole_page(monkeypatch) -> None:
    body = _lines(200)
    _serve(monkeypatch, _status(200, body, **{"content-type": "text/plain"}))
    assert web.read("https://example.com/") == body


def test_a_short_page_is_shown_whole() -> None:
    assert web.window("short page", 0, 100) == "short page"


def test_a_long_page_ends_at_a_line_with_the_offset_to_read_on() -> None:
    text = _lines(100)  # 10,000 characters
    shown = web.window(text, 0, 1_050)
    body, _, footer = shown.rpartition("\n\n")
    assert body.endswith("x")  # whole lines only
    assert body == text[:1_000].rstrip()
    assert "of 9,999" in footer
    assert "offset 1000" in footer


def test_reading_on_continues_where_the_last_part_ended() -> None:
    text = _lines(100)
    shown = web.window(text, 1_000, 1_000)
    assert shown.startswith("[Continuing from character 1,000 of 9,999.]")
    assert "0010" in shown.split("\n\n", 1)[1][:4]


def test_the_last_part_has_no_offset_to_read_on() -> None:
    text = _lines(10)
    shown = web.window(text, 500, 5_000)
    assert "offset" not in shown.rsplit("\n\n", 1)[-1]
    assert shown.endswith(text[-20:])


def test_an_offset_past_the_end_is_an_error() -> None:
    with pytest.raises(web.WebError, match="past the end"):
        web.window("tiny", 10, 100)


def test_reddit_threads_get_a_bigger_page() -> None:
    assert web.page_chars(
        "https://www.reddit.com/r/FoodNYC/comments/x/"
    ) > web.page_chars("https://example.com/")


# -- finding passages ---------------------------------------------------------------

MENU = """\
Antipasti
Burrata with tomatoes (v)
Meatballs al forno

Pasta
Cacio e pepe, vegetarian
Spaghetti alla carbonara with guanciale
Eggplant parmigiana, a vegetarian classic
"""


def test_matching_passages_come_with_offsets() -> None:
    found = web.find_passages(MENU, "vegetarian")
    assert "2 passages" in found
    for line in ("Cacio e pepe", "Eggplant parmigiana"):
        assert f"@{MENU.index(line)}: {line}" in found
    assert "Meatballs" not in found


def test_passages_matching_more_words_rank_first() -> None:
    found = web.find_passages(MENU, "eggplant vegetarian", limit=1)
    assert "Eggplant parmigiana" in found
    assert "Cacio" not in found


def test_matching_ignores_case() -> None:
    assert "Burrata" in web.find_passages(MENU, "BURRATA")


def test_no_match_says_so_and_suggests_what_to_try() -> None:
    found = web.find_passages(MENU, "vegan")
    assert "No passages match" in found
    assert "fetch_url" in found


def test_long_lines_are_cut_to_the_match() -> None:
    text = "a" * 5_000 + " vegetarian lasagna " + "b" * 5_000
    found = web.find_passages(text, "vegetarian")
    assert len(found) < 1_500
    assert "vegetarian lasagna" in found
    offset = int(found.split("\n@", 1)[1].split(":", 1)[0])
    assert 4_000 < offset <= 5_001
