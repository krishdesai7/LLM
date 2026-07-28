from __future__ import annotations

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
    assert web_toolbox().names == ["web_search", "fetch_url"]


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
    assert box.names == ["web_search", "fetch_url", "now"]
    assert box.call("now", "{}") == "12"
