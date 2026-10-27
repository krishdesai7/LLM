"""Malformed tool calls: broken arguments, and calls written out as text."""

from __future__ import annotations

import json

import pytest

from llm.repair import canonical_name, looks_like_call, parse_arguments, recover_calls

NAMES = ("web_search", "fetch_url")


# -- arguments ----------------------------------------------------------------


def test_clean_arguments_need_no_repair() -> None:
    assert parse_arguments('{"url": "https://a.com"}') == (
        {"url": "https://a.com"},
        False,
    )


def test_empty_arguments_are_an_empty_object() -> None:
    assert parse_arguments("  ") == ({}, False)


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"url": "https://a.com"}\n```',
        'Here you go: {"url": "https://a.com"}',
        '{"url": "https://a.com",}',
        "{'url': 'https://a.com'}",
        json.dumps(json.dumps({"url": "https://a.com"})),
    ],
    ids=["fenced", "preamble", "trailing-comma", "single-quotes", "double-encoded"],
)
def test_common_damage_is_repaired(text: str) -> None:
    assert parse_arguments(text) == ({"url": "https://a.com"}, True)


def test_hopeless_arguments_raise_with_the_json_error() -> None:
    with pytest.raises(ValueError, match="Expecting"):
        parse_arguments('{"url": "https://a.com"')


def test_a_non_object_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_arguments("[1, 2]")


# -- names --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "name"),
    [
        ("fetch_url", "fetch_url"),
        ("functions.fetch_url", "fetch_url"),
        ("Fetch-URL", "fetch_url"),
        ("browse", None),
    ],
)
def test_names_are_matched_up_to_formatting(raw: str, name: str | None) -> None:
    assert canonical_name(raw, NAMES) == name


# -- calls written as text ------------------------------------------------------


def _recovered(text: str) -> tuple[list[tuple[str, dict]], str]:
    found = recover_calls(text, NAMES)
    assert found is not None, "expected a call to be recovered"
    return [(n, json.loads(a)) for n, a in found.calls], found.rest


def test_a_fenced_call_at_the_end_is_recovered() -> None:
    text = (
        "Found the thread. Now the menu.\n\n```json\n"
        '{"name": "fetch_url", "arguments": {"url": "https://a.com/menu"}}\n```'
    )
    calls, rest = _recovered(text)
    assert calls == [("fetch_url", {"url": "https://a.com/menu"})]
    assert rest == "Found the thread. Now the menu."


def test_a_bare_openai_shaped_call_is_recovered() -> None:
    text = json.dumps(
        {
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"query": "x"}'},
        }
    )
    calls, rest = _recovered(text)
    assert calls == [("web_search", {"query": "x"})]
    assert rest == ""


def test_a_list_of_calls_is_recovered() -> None:
    text = json.dumps(
        {
            "tool_calls": [
                {"name": "fetch_url", "parameters": {"url": "https://a.com"}},
                {"tool": "web_search", "args": {"query": "b"}},
            ]
        }
    )
    calls, _ = _recovered(text)
    assert [n for n, _ in calls] == ["fetch_url", "web_search"]


def test_kimi_markers_are_recovered() -> None:
    text = (
        "Checking it.<|tool_calls_section_begin|><|tool_call_begin|>"
        'functions.fetch_url:0<|tool_call_argument_begin|>{"url": "https://a.com"}'
        "<|tool_call_end|><|tool_calls_section_end|>"
    )
    calls, rest = _recovered(text)
    assert calls == [("fetch_url", {"url": "https://a.com"})]
    assert rest == "Checking it."


def test_deepseek_markers_are_recovered() -> None:
    bar, sp = "\uff5c", "\u2581"  # DeepSeek's full-width bar and block
    text = (
        f"<{bar}tool{sp}calls{sp}begin{bar}><{bar}tool{sp}call{sp}begin{bar}>"
        f"function<{bar}tool{sp}sep{bar}>web_search\n"
        f'```json\n{{"query": "x"}}\n```'
        f"<{bar}tool{sp}call{sp}end{bar}><{bar}tool{sp}calls{sp}end{bar}>"
    )
    calls, rest = _recovered(text)
    assert calls == [("web_search", {"query": "x"})]
    assert rest == ""


def test_qwen_tool_call_tags_are_recovered() -> None:
    text = (
        '<tool_call>\n{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>'
    )
    calls, _ = _recovered(text)
    assert calls == [("web_search", {"query": "x"})]


def test_a_function_call_spelling_is_recovered() -> None:
    calls, rest = _recovered('Next:\nfetch_url({"url": "https://a.com"})')
    assert calls == [("fetch_url", {"url": "https://a.com"})]
    assert rest == "Next:"


def test_json_in_the_middle_of_an_answer_is_left_alone() -> None:
    text = (
        'You would call it like {"name": "fetch_url", "arguments": {}} and then '
        "read the page. That's the whole answer."
    )
    assert recover_calls(text, NAMES) is None


def test_unknown_tools_are_not_recovered() -> None:
    assert recover_calls('{"name": "browse", "arguments": {}}', NAMES) is None


def test_plain_answers_are_not_calls() -> None:
    assert recover_calls("Via Carota has three vegetarian pastas.", NAMES) is None
    assert not looks_like_call("Via Carota has three vegetarian pastas.", NAMES)


def test_a_garbled_attempt_still_looks_like_a_call() -> None:
    assert looks_like_call("<|tool_call_begin|>functions.fetch_url:0 {oops", NAMES)
    assert looks_like_call(
        '```json\n{"name": "fetch_url", "arguments": {"url": ', NAMES
    )
