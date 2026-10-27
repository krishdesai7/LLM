from __future__ import annotations

import json
from typing import Any

from llm import web
from llm.tools import Tool, ToolBox, web_toolbox


def _box(fn: Any) -> ToolBox:
    return ToolBox(
        [
            Tool("t", "a tool", {"type": "object", "properties": {}}, fn),
        ]
    )


def test_schema_is_openai_shaped() -> None:
    schema = web_toolbox().schemas()[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "web_search"
    assert "query" in schema["function"]["parameters"]["properties"]


def test_default_toolbox_offers_search_and_fetch() -> None:
    assert web_toolbox().names == ["web_search", "fetch_url", "find_in_page"]


def test_arguments_are_parsed_from_json() -> None:
    seen: dict[str, Any] = {}

    def run(**kwargs: Any) -> str:
        seen.update(kwargs)
        return "ok"

    assert _box(run).call("t", '{"a": 1, "b": "two"}') == "ok"
    assert seen == {"a": 1, "b": "two"}


def test_empty_arguments_are_allowed() -> None:
    assert _box(lambda: "ok").call("t", "") == "ok"


def test_unknown_tool_returns_an_error_string() -> None:
    assert "no tool named" in _box(lambda: "ok").call("nope", "{}")


def test_malformed_json_returns_an_error_string() -> None:
    assert "not valid JSON" in _box(lambda: "ok").call("t", "{oh no")


def test_non_object_arguments_are_rejected() -> None:
    assert "must be a JSON object" in _box(lambda: "ok").call("t", "[1, 2]")


def test_wrong_arguments_return_an_error_string() -> None:
    assert "wrong arguments" in _box(lambda: "ok").call("t", '{"nope": 1}')


def test_web_errors_are_surfaced_cleanly() -> None:
    def run(**_kwargs: Any) -> str:
        raise web.WebError("refusing to fetch localhost")

    assert _box(run).call("t", "{}") == "Error: refusing to fetch localhost"


def test_unexpected_errors_are_contained() -> None:
    def run(**_kwargs: Any) -> str:
        raise ZeroDivisionError("oops")

    assert _box(run).call("t", "{}") == "Error: ZeroDivisionError: oops"


def test_an_empty_toolbox_is_falsy() -> None:
    assert not ToolBox()
    assert web_toolbox()


def test_a_box_can_be_extended_with_your_own_tools() -> None:
    clock = Tool("now", "the time", {"type": "object", "properties": {}}, lambda: "12")
    box = ToolBox([*web_toolbox().tools, clock])
    assert box.names == ["web_search", "fetch_url", "find_in_page", "now"]
    assert box.call("now", "{}") == "12"


# -- remembering what was already tried ---------------------------------------


def _counting(outcome: Any) -> tuple[ToolBox, list[dict[str, Any]]]:
    """A box whose ``fetch_url`` records each run and returns or raises ``outcome``."""
    runs: list[dict[str, Any]] = []

    def run(**kwargs: Any) -> str:
        runs.append(kwargs)
        result = outcome(kwargs["url"]) if callable(outcome) else outcome
        if isinstance(result, Exception):
            raise result
        return result

    params = {"type": "object", "properties": {"url": {"type": "string"}}}
    return ToolBox([Tool("fetch_url", "fetch", params, run)]), runs


def _blocked(url: str) -> web.FetchError:
    return web.FetchError("blocked", url, status=403)


def test_a_failed_call_is_not_run_again() -> None:
    box, runs = _counting(_blocked)
    first = box.call("fetch_url", '{"url": "https://a.com/menu"}')
    second = box.call("fetch_url", '{"url": "https://a.com/menu"}')
    assert len(runs) == 1
    assert first.startswith("Error:")
    assert "not run" in second
    assert "call #1" in second
    assert "HTTP 403" in second


def test_a_retryable_failure_gets_one_retry() -> None:
    error = web.FetchError("server_error", "https://a.com/", status=502)
    box, runs = _counting(error)
    for _ in range(3):
        box.call("fetch_url", '{"url": "https://a.com/"}')
    assert len(runs) == 2


def test_a_repeated_success_is_answered_from_memory() -> None:
    box, runs = _counting("the menu")
    assert box.call("fetch_url", '{"url": "https://a.com/menu"}') == "the menu"
    again = box.call("fetch_url", '{"url": "https://a.com/menu#mains"}')
    assert len(runs) == 1
    assert "same call as #1" in again
    assert again.endswith("the menu")  # in full, as the original may be shortened


def test_trivial_differences_count_as_the_same_call() -> None:
    box, runs = _counting("ok")
    box.call("fetch_url", '{"url": "https://a.com/menu/"}')
    box.call("fetch_url", '{"url": "  https://a.com/menu"}')
    assert len(runs) == 1


def test_queries_are_compared_without_case_or_spacing() -> None:
    runs: list[str] = []
    params = {"type": "object", "properties": {"query": {"type": "string"}}}
    box = ToolBox(
        [Tool("web_search", "s", params, lambda query: runs.append(query) or "r")]
    )
    box.call("web_search", '{"query": "Via Carota  menu"}')
    box.call("web_search", '{"query": "via carota menu"}')
    box.call("web_search", '{"query": "via carota reviews"}')
    assert runs == ["Via Carota  menu", "via carota reviews"]


def test_a_host_that_keeps_blocking_is_skipped() -> None:
    box, runs = _counting(_blocked)
    box.call("fetch_url", '{"url": "https://www.a.com/one"}')
    box.call("fetch_url", '{"url": "https://a.com/two"}')
    third = box.call("fetch_url", '{"url": "https://a.com/three"}')
    assert len(runs) == 2
    assert "not run" in third
    assert "a.com" in third
    box.call("fetch_url", '{"url": "https://b.com/one"}')
    assert len(runs) == 3


def test_reset_forgets_earlier_calls() -> None:
    box, runs = _counting(_blocked)
    box.call("fetch_url", '{"url": "https://a.com/"}')
    box.reset()
    box.call("fetch_url", '{"url": "https://a.com/"}')
    assert len(runs) == 2


# -- helping a model that drifts ------------------------------------------------


def test_repaired_arguments_run_and_are_counted() -> None:
    box, runs = _counting("ok")
    assert box.call("fetch_url", '```json\n{"url": "https://a.com"}\n```') == "ok"
    assert runs == [{"url": "https://a.com"}]
    assert box.events["repaired"] == 1


def test_bad_json_errors_show_how_to_call_the_tool() -> None:
    message = web_toolbox().call("fetch_url", '{"url": "https://a.com"')
    assert "not valid JSON" in message
    assert '"url" (string, required)' in message
    assert 'Example: {"url": "https://example.com/page"}' in message


def test_wrong_argument_names_show_the_parameters() -> None:
    message = web_toolbox().call("fetch_url", '{"link": "https://a.com"}')
    assert "wrong arguments" in message
    assert '"url" (string, required)' in message


def test_a_formatting_variant_of_a_tool_name_still_runs() -> None:
    box, runs = _counting("ok")
    assert box.call("functions.fetch_url", '{"url": "https://a.com"}') == "ok"
    assert len(runs) == 1


def test_an_unknown_tool_suggests_the_closest_one() -> None:
    message = web_toolbox().call("fetch", '{"url": "https://a.com"}')
    assert "no tool named 'fetch'" in message
    assert "Did you mean 'fetch_url'?" in message


def test_refused_repeats_are_counted() -> None:
    box, _ = _counting(_blocked)
    box.call("fetch_url", '{"url": "https://a.com/"}')
    box.call("fetch_url", '{"url": "https://a.com/"}')
    assert box.events["refused"] == 1


def test_a_run_of_failures_says_to_change_tack() -> None:
    box, _ = _counting(lambda url: web.FetchError("not_found", url, status=404))
    outputs = [
        box.call("fetch_url", json.dumps({"url": f"https://a.com/{i}"}))
        for i in range(3)
    ]
    assert "No progress" not in outputs[1]
    assert "No progress" in outputs[2]
    assert "#1" in outputs[2]
    assert "#3" in outputs[2]
    assert "different kind of source" in outputs[2]
    assert box.events["stalled"] == 1


def test_a_success_resets_the_run_of_failures() -> None:
    def outcome(url: str) -> Any:
        return (
            "ok" if url.endswith("ok") else web.FetchError("not_found", url, status=404)
        )

    box, _ = _counting(outcome)
    for path in ("1", "2", "ok", "3", "4"):
        last = box.call("fetch_url", json.dumps({"url": f"https://a.com/{path}"}))
    assert "No progress" not in last


def test_a_new_turn_resets_counts_but_not_memory() -> None:
    box, runs = _counting(_blocked)
    box.call("fetch_url", '{"url": "https://a.com/"}')
    box.call("fetch_url", '{"url": "https://a.com/"}')
    box.start_turn()
    assert not box.events
    box.call("fetch_url", '{"url": "https://a.com/"}')
    assert len(runs) == 1


# -- paging through cached pages ------------------------------------------------------


def _reads(monkeypatch, text: str) -> list[str]:
    """Stand in for the network: every page is ``text``; returns the URLs read."""
    reads: list[str] = []

    def read(url: str) -> str:
        reads.append(url)
        return text

    monkeypatch.setattr(web, "read", read)
    return reads


LONG = "".join(f"line {i:03d} " + "x" * 90 + "\n" for i in range(200))


def test_the_web_tools_include_find_in_page() -> None:
    assert "find_in_page" in web_toolbox().names


def test_reading_on_through_a_page_fetches_it_once(monkeypatch) -> None:
    reads = _reads(monkeypatch, LONG)
    box = web_toolbox()
    first = box.call("fetch_url", '{"url": "https://a.com/menu"}')
    offset = int(first.rsplit("offset ", 1)[1].split(",")[0].rstrip(".]"))
    second = box.call(
        "fetch_url", json.dumps({"url": "https://a.com/menu", "offset": offset})
    )
    assert len(reads) == 1
    assert "line 000" in first
    assert second.startswith("[Continuing from character")
    assert f"line {offset // 100:03d}" in second


def test_find_in_page_uses_the_cached_page(monkeypatch) -> None:
    reads = _reads(monkeypatch, LONG + "Eggplant parmigiana (vegetarian)\n")
    box = web_toolbox()
    box.call("fetch_url", '{"url": "https://a.com/menu"}')
    found = box.call(
        "find_in_page", '{"url": "https://a.com/menu#top", "query": "eggplant"}'
    )
    assert len(reads) == 1
    assert "Eggplant parmigiana" in found


def test_find_in_page_fetches_a_page_it_has_not_seen(monkeypatch) -> None:
    reads = _reads(monkeypatch, "Burrata (v)")
    found = web_toolbox().call(
        "find_in_page", '{"url": "https://b.com", "query": "burrata"}'
    )
    assert reads == ["https://b.com"]
    assert "Burrata" in found


def test_each_toolbox_has_its_own_cache(monkeypatch) -> None:
    reads = _reads(monkeypatch, "page")
    web_toolbox().call("fetch_url", '{"url": "https://a.com"}')
    web_toolbox().call("fetch_url", '{"url": "https://a.com"}')
    assert len(reads) == 2


def test_cached_pages_expire(monkeypatch) -> None:
    reads = _reads(monkeypatch, LONG)
    clock = [1_000.0]
    monkeypatch.setattr(web, "_now", lambda: clock[0])
    pages = web.Pages()
    pages.text("https://a.com")
    clock[0] += web.PAGE_TTL + 1
    pages.text("https://a.com")
    assert len(reads) == 2
