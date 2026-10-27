"""The model asks; we run the tool; the answer comes back in a second round."""

from __future__ import annotations

import json
from typing import Any

from conftest import calls_of, text_chunk, tool_call_chunks

from llm.chat import REMIND_EVERY, TOOL_BUDGET
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


def _searches(count: int) -> list[list[Any]]:
    return [
        tool_call_chunks("web_search", json.dumps({"query": f"q{i}"}))
        for i in range(count)
    ]


def test_the_default_budget_leaves_room_for_research() -> None:
    assert TOOL_BUDGET >= 30


def test_a_spent_budget_asks_for_a_final_answer(make_session: Any) -> None:
    rounds = [*_searches(3), [text_chunk("Here is what I found.")]]
    session = make_session(rounds=rounds, tools=_echo_box(), tool_budget=3)
    deltas = list(session.stream("research this"))
    text = "".join(d.text for d in deltas if d.kind == "text")
    assert text == "Here is what I found."
    requests = calls_of(session)
    assert len(requests) == 4
    assert [r["tool_choice"] for r in requests] == ["auto", "auto", "auto", "none"]
    assert any("final answer" in d.text for d in deltas if d.kind == "notice")


def test_the_last_tool_result_says_the_budget_is_spent(make_session: Any) -> None:
    rounds = [*_searches(2), [text_chunk("done")]]
    session = make_session(rounds=rounds, tools=_echo_box(), tool_budget=2)
    list(session.stream("hi"))
    last_tool = [m for m in session.messages if m.role == "tool"][-1]
    assert "no more tools" in last_tool.content


def test_results_warn_as_the_budget_runs_low(make_session: Any) -> None:
    rounds = [*_searches(1), [text_chunk("done")]]
    session = make_session(rounds=rounds, tools=_echo_box(), tool_budget=4)
    list(session.stream("hi"))
    tool = next(m for m in session.messages if m.role == "tool")
    assert "3 of 4 tool calls left" in tool.content


def test_calls_past_the_budget_are_answered_but_not_run(make_session: Any) -> None:
    received: list[dict[str, Any]] = []
    two_at_once = tool_call_chunks("web_search", '{"query": "a"}', call_id="c1")
    second = tool_call_chunks("web_search", '{"query": "b"}', call_id="c2")
    for chunk in second:
        for fragment in chunk.choices[0].delta.tool_calls or []:
            fragment.index = 1
    rounds = [two_at_once + second, [text_chunk("done")]]
    session = make_session(rounds=rounds, tools=_echo_box(received), tool_budget=1)
    list(session.stream("hi"))
    assert received == [{"query": "a"}]
    answers = {m.tool_call_id: m.content for m in session.messages if m.role == "tool"}
    assert set(answers) == {"c1", "c2"}
    assert "not run" in answers["c2"]


def test_a_model_that_ignores_the_final_round_is_stopped(make_session: Any) -> None:
    forever = _searches(1)
    session = make_session(rounds=forever, tools=_echo_box(), tool_budget=3)
    notices = [d for d in session.stream("hi") if d.kind == "notice"]
    assert len(calls_of(session)) == 4
    assert any("stopped" in notice.text for notice in notices)
    assert session.messages[-1].role == "tool"


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


# -- calls written as text ------------------------------------------------------

LEAKED = (
    'Checking.\n```json\n{"name": "web_search", "arguments": {"query": "leaked"}}\n```'
)


def test_a_call_written_as_text_is_run(make_session: Any) -> None:
    received: list[dict[str, Any]] = []
    rounds = [[text_chunk(LEAKED)], [text_chunk("Found it.")]]
    session = make_session(rounds=rounds, tools=_echo_box(received))
    deltas = list(session.stream("hi"))
    assert received == [{"query": "leaked"}]
    assert "".join(d.text for d in deltas if d.kind == "text").endswith("Found it.")
    assert any("recovered 1 tool call" in d.text for d in deltas if d.kind == "notice")


def test_the_recovered_call_is_rewritten_as_a_real_one(make_session: Any) -> None:
    rounds = [[text_chunk(LEAKED)], [text_chunk("Found it.")]]
    session = make_session(rounds=rounds, tools=_echo_box())
    list(session.stream("hi"))
    rewritten = session.messages[1]
    assert rewritten.content == "Checking."
    assert rewritten.tool_calls[0].name == "web_search"
    sent = calls_of(session)[1]["messages"]
    assert sent[-2]["tool_calls"][0]["function"]["name"] == "web_search"
    assert sent[-1]["tool_call_id"] == rewritten.tool_calls[0].id


def test_a_garbled_call_is_asked_for_again_once(make_session: Any) -> None:
    garbled = '```json\n{"name": "web_search", "arguments": {"query": '
    rounds = [[text_chunk(garbled)], [text_chunk("Plain answer.")]]
    session = make_session(rounds=rounds, tools=_echo_box())
    deltas = list(session.stream("hi"))
    requests = calls_of(session)
    assert len(requests) == 2
    retry = requests[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": garbled.strip()}
    assert retry[-1]["role"] == "user"
    assert "tool-calling interface" in retry[-1]["content"]
    assert [m.role for m in session.messages] == ["user", "assistant"]
    assert session.messages[-1].content == "Plain answer."
    assert any("asking again" in d.text for d in deltas if d.kind == "notice")


def test_a_second_garbled_call_is_taken_as_the_answer(make_session: Any) -> None:
    garbled = "<|tool_call_begin|>functions.web_search:0 {oops"
    session = make_session(rounds=[[text_chunk(garbled)]], tools=_echo_box())
    list(session.stream("hi"))
    assert len(calls_of(session)) == 2
    assert session.messages[-1].content == garbled


def test_nothing_is_recovered_in_the_final_round(make_session: Any) -> None:
    rounds = [*_searches(1), [text_chunk(LEAKED)]]
    session = make_session(rounds=rounds, tools=_echo_box(), tool_budget=1)
    list(session.stream("hi"))
    assert len(calls_of(session)) == 2
    assert session.messages[-1].content == LEAKED


# -- keeping the model on track ---------------------------------------------------


def test_the_rules_are_restated_every_few_calls(make_session: Any) -> None:
    rounds = [*_searches(REMIND_EVERY), [text_chunk("done")]]
    session = make_session(rounds=rounds, tools=_echo_box())
    list(session.stream("hi"))
    results = [m.content for m in session.messages if m.role == "tool"]
    assert "Reminder:" in results[REMIND_EVERY - 1]
    assert not any("Reminder:" in r for r in results[: REMIND_EVERY - 1])


def test_a_turn_ends_with_a_count_of_what_the_harness_fixed(make_session: Any) -> None:
    rounds = [[text_chunk(LEAKED)], [text_chunk("Found it.")]]
    session = make_session(rounds=rounds, tools=_echo_box())
    notices = [d.text for d in session.stream("hi") if d.kind == "notice"]
    assert notices[-1].startswith("harness:")
    assert "1 tool call recovered from text" in notices[-1]
    assert session.last_events["recovered"] == 1


def test_a_clean_turn_has_no_count(make_session: Any) -> None:
    session = make_session(rounds=_search_then_answer(), tools=_echo_box())
    assert not [d for d in session.stream("hi") if d.kind == "notice"]
