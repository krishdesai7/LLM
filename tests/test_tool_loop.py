"""The model asks; we run the tool; the answer comes back in a second round."""

from __future__ import annotations

import json
from typing import Any

from conftest import calls_of, text_chunk, tool_call_chunks

from llm.chat import MAX_TOOL_ROUNDS
from llm.tools import Tool, ToolBox


def _echo_box(calls: list[dict[str, Any]] | None = None) -> ToolBox:
    def echo(**kwargs: Any) -> str:
        if calls is not None:
            calls.append(kwargs)
        return f"echo: {kwargs.get('query', '')}"

    return ToolBox(
        [
            Tool(
                name="web_search",
                description="search",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                run=echo,
            )
        ]
    )


def _search_then_answer() -> list[list[Any]]:
    return [
        tool_call_chunks("web_search", json.dumps({"query": "python 3.14"})),
        [text_chunk("It shipped.")],
    ]


def test_tools_are_advertised_when_present(make_session: Any) -> None:
    session = make_session(tools=_echo_box())
    list(session.stream("hi"))
    sent = calls_of(session)[0]
    assert sent["tool_choice"] == "auto"
    assert sent["tools"][0]["function"]["name"] == "web_search"


def test_a_tool_call_triggers_a_second_round(make_session: Any) -> None:
    session = make_session(rounds=_search_then_answer(), tools=_echo_box())
    text = "".join(d.text for d in session.stream("what's new") if d.kind == "text")
    assert text == "It shipped."
    assert len(calls_of(session)) == 2


def test_the_tool_actually_runs_with_parsed_arguments(make_session: Any) -> None:
    received: list[dict[str, Any]] = []
    session = make_session(rounds=_search_then_answer(), tools=_echo_box(received))
    list(session.stream("what's new"))
    assert received == [{"query": "python 3.14"}]


def test_streamed_argument_fragments_are_reassembled(make_session: Any) -> None:
    session = make_session(rounds=_search_then_answer(), tools=_echo_box())
    deltas = [d for d in session.stream("what's new") if d.kind == "tool_call"]
    assert deltas[0].text == 'web_search({"query": "python 3.14"})'


def test_history_carries_the_call_and_its_result(make_session: Any) -> None:
    session = make_session(rounds=_search_then_answer(), tools=_echo_box())
    list(session.stream("what's new"))
    roles = [m.role for m in session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]

    assistant = session.messages[1]
    assert assistant.tool_calls[0].name == "web_search"
    tool_message = session.messages[2]
    assert tool_message.tool_call_id == assistant.tool_calls[0].id
    assert tool_message.content == "echo: python 3.14"


def test_the_tool_result_is_sent_back_to_the_model(make_session: Any) -> None:
    session = make_session(rounds=_search_then_answer(), tools=_echo_box())
    list(session.stream("what's new"))
    second_request = calls_of(session)[1]["messages"]
    assert second_request[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "echo: python 3.14",
    }
    assert second_request[-2]["tool_calls"][0]["function"]["name"] == "web_search"


def test_usage_is_summed_across_rounds(make_session: Any, chunks: list[Any]) -> None:
    session = make_session(rounds=[chunks, chunks], tools=_echo_box())
    list(session.stream("hi"))
    assert session.last_usage is not None
    assert session.last_usage.total == 10  # one round: no tool call was made


def test_runaway_tool_loops_are_capped(make_session: Any) -> None:
    forever = [tool_call_chunks("web_search", '{"query": "x"}')]
    session = make_session(rounds=forever, tools=_echo_box())
    notices = [d for d in session.stream("hi") if d.kind == "notice"]
    assert len(calls_of(session)) == MAX_TOOL_ROUNDS
    assert "stopped after" in notices[0].text


def test_a_failing_tool_is_reported_to_the_model_not_raised(make_session: Any) -> None:
    def boom(**_kwargs: Any) -> str:
        raise RuntimeError("network on fire")

    box = ToolBox(
        [Tool("web_search", "search", {"type": "object", "properties": {}}, boom)]
    )
    session = make_session(rounds=_search_then_answer(), tools=box)
    list(session.stream("hi"))
    assert "network on fire" in session.messages[2].content
    assert len(calls_of(session)) == 2


def test_transcript_records_tool_use(make_session: Any) -> None:
    session = make_session(rounds=_search_then_answer(), tools=_echo_box())
    list(session.stream("what's new"))
    assert "called `web_search(" in session.transcript()
