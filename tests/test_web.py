from __future__ import annotations

import httpx
import pytest

from llm import web

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

    def client(**kwargs):
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


def test_http_errors_propagate_as_httpx_errors(monkeypatch) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="nope")

    _serve(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError):
        web.fetch("https://example.com/")


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
