from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from conftest import (
    PROVIDER,
    FakeChoice,
    FakeChunk,
    FakeDelta,
    calls_of,
    text_chunk,
    tool_call_chunks,
)

from llm import chat
from llm.chat import Message, ToolCall
from llm.registry import ModelRef, Provider
from llm.tools import WEB_SEARCH, Tool, ToolBox, web_toolbox


def test_stream_yields_reasoning_and_text(make_session: Any) -> None:
    session = make_session()
    deltas = list(session.stream("hi"))
    assert [(d.kind, d.text) for d in deltas] == [
        ("reasoning", "hmm"),
        ("text", "Hello"),
        ("text", ", world"),
    ]


def test_stream_records_history_and_usage(make_session: Any) -> None:
    session = make_session()
    list(session.stream("hi"))
    assert [(m.role, m.content) for m in session.messages] == [
        ("user", "hi"),
        ("assistant", "Hello, world"),
    ]
    assert session.last_usage is not None
    assert session.last_usage.total == 10


def test_reasoning_is_not_fed_back_into_history(make_session: Any) -> None:
    session = make_session()
    list(session.stream("hi"))
    assert "hmm" not in session.messages[-1].content


def test_send_returns_just_the_reply_text(make_session: Any) -> None:
    session = make_session()
    assert session.send("hi") == "Hello, world"


def test_no_tools_are_advertised_by_default(make_session: Any) -> None:
    session = make_session()
    list(session.stream("hi"))
    assert "tools" not in calls_of(session)[0]


@pytest.fixture
def frozen_today(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat, "_today", lambda: date(2026, 10, 18))


@pytest.mark.usefixtures("frozen_today")
def test_preamble_with_the_date_leads_every_payload(make_session: Any) -> None:
    session = make_session()
    list(session.stream("hi"))
    sent = calls_of(session)[0]["messages"]
    assert sent[0]["role"] == "system"
    assert "Today is Sunday, October 18, 2026." in sent[0]["content"]
    assert sent[1] == {"role": "user", "content": "hi"}


def test_user_system_prompt_follows_the_preamble(make_session: Any) -> None:
    session = make_session(system="be terse")
    prompt = session.system_prompt()
    assert prompt.startswith("Today is")
    assert prompt.endswith("\n\nbe terse")


def test_web_guidance_only_while_tools_are_on(make_session: Any) -> None:
    session = make_session()
    assert "web_search" not in session.system_prompt()
    session.tools = ToolBox([WEB_SEARCH])
    assert "web_search" in session.system_prompt()


def test_params_are_forwarded(make_session: Any) -> None:
    session = make_session(params={"temperature": 0.2})
    list(session.stream("hi"))
    assert calls_of(session)[0]["temperature"] == pytest.approx(0.2)


def test_multi_turn_context_is_accumulated(make_session: Any) -> None:
    session = make_session()
    list(session.stream("first"))
    list(session.stream("second"))
    sent = calls_of(session)[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]


def test_clear_resets_the_conversation(make_session: Any) -> None:
    session = make_session()
    list(session.stream("hi"))
    session.clear()
    assert session.messages == []
    assert session.last_usage is None


def test_abandoned_stream_still_saves_what_arrived(make_session: Any) -> None:
    session = make_session()
    stream = session.stream("hi")
    next(stream)
    next(stream)
    stream.close()
    assert session.messages[-1].content == "Hello"


def test_empty_chunks_are_tolerated(make_session: Any) -> None:
    session = make_session(chunks=[FakeChunk()])
    assert list(session.stream("hi")) == []
    assert session.messages[-1].content == ""


def test_transcript_is_markdown(make_session: Any) -> None:
    session = make_session(system="be terse")
    list(session.stream("hi"))
    transcript = session.transcript()
    assert transcript.startswith("# fake/fake-1")
    assert "## system" in transcript
    assert "Hello, world" in transcript


def test_assistant_message_keeps_reasoning(make_session: Any) -> None:
    session = make_session()
    list(session.stream("hi"))
    assert session.messages[-1].reasoning == "hmm"


def test_reasoning_seconds_spans_first_reasoning_to_first_text(
    make_session: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = iter([10.0, 14.5, 99.0, 99.0])
    monkeypatch.setattr(chat, "_now", lambda: next(clock))
    session = make_session()
    list(session.stream("hi"))
    assert session.messages[-1].reasoning_seconds == pytest.approx(4.5)


def test_reasoning_is_never_sent_back_to_the_api(make_session: Any) -> None:
    session = make_session()
    list(session.stream("hi"))
    list(session.stream("again"))
    sent = calls_of(session)[1]["messages"]
    assert {"role": "assistant", "content": "Hello, world"} in sent
    assert all("reasoning" not in message for message in sent)


def test_message_without_reasoning_has_zero_seconds(make_session: Any) -> None:
    session = make_session(chunks=[text_chunk("plain")])
    list(session.stream("hi"))
    assert session.messages[-1].reasoning == ""
    assert session.messages[-1].reasoning_seconds == pytest.approx(0.0)


def test_web_guidance_scales_effort_and_names_the_budget(make_session: Any) -> None:
    session = make_session(tools=web_toolbox(), tool_budget=25)
    prompt = session.system_prompt()
    assert "One or two targeted searches are usually enough" not in prompt
    assert "up to 25" in prompt
    assert "notes" in prompt


# -- shrinking old tool results ---------------------------------------------


def _researched(*sizes: int) -> list[Message]:
    """A turn that fetched one page per size, the model reading each in turn."""
    messages = [Message("user", "research it")]
    for index, size in enumerate(sizes):
        call = ToolCall(
            f"c{index}", "fetch_url", json.dumps({"url": f"https://p{index}.com"})
        )
        messages += [
            Message("assistant", "", tool_calls=[call]),
            Message("tool", f"page {index} " + "x" * size, tool_call_id=call.id),
        ]
    return messages


def _sent_tools(session: Any) -> list[str]:
    return [m["content"] for m in session.payload() if m["role"] == "tool"]


def test_old_long_results_are_shortened_in_what_is_sent(make_session: Any) -> None:
    session = make_session(tools=web_toolbox())
    session.messages = _researched(50_000, 50_000, 50_000)
    first, second, third = _sent_tools(session)
    assert len(first) < 1_000
    assert "fetch_url" in first
    assert "https://p0.com" in first
    assert "again" in first
    assert first.startswith("[")
    assert len(third) > 50_000
    assert len(second) > 50_000


def test_history_itself_keeps_every_result_whole(make_session: Any) -> None:
    session = make_session(tools=web_toolbox())
    session.messages = _researched(50_000, 50_000, 50_000)
    session.payload()
    assert all(len(m.content) > 50_000 for m in session.messages if m.role == "tool")


def test_short_results_are_never_shortened(make_session: Any) -> None:
    session = make_session(tools=web_toolbox())
    session.messages = _researched(500, 90_000, 90_000)
    assert _sent_tools(session)[0] == session.messages[2].content


def test_results_the_model_has_not_read_yet_are_kept(make_session: Any) -> None:
    session = make_session(tools=web_toolbox())
    calls = [ToolCall(f"c{i}", "fetch_url", "{}") for i in range(3)]
    session.messages = [
        Message("user", "go"),
        Message("assistant", "", tool_calls=calls),
        *(Message("tool", "y" * 60_000, tool_call_id=c.id) for c in calls),
    ]
    assert all(len(sent) == 60_000 for sent in _sent_tools(session))


# -- sending reasoning back ---------------------------------------------------


def _replaying(session: Any, replay: str) -> Any:
    provider = Provider(
        PROVIDER.name,
        PROVIDER.env_var,
        PROVIDER.base_url,
        reasoning_replay=(("fake-", replay),),  # ty: ignore[invalid-argument-type]
    )
    session.ref = ModelRef(provider, "fake-1")
    return session


def _two_turns() -> list[Message]:
    """An earlier turn that used a tool, then a new question in progress."""
    old = ToolCall("c0", "web_search", '{"query": "old"}')
    new = ToolCall("c1", "web_search", '{"query": "new"}')
    return [
        Message("user", "first"),
        Message("assistant", "", tool_calls=[old], reasoning="plan A"),
        Message("tool", "result A", tool_call_id="c0"),
        Message("assistant", "answer A", reasoning="wrap A"),
        Message("user", "second"),
        Message("assistant", "", tool_calls=[new], reasoning="plan B"),
        Message("tool", "result B", tool_call_id="c1"),
    ]


def _sent_reasoning(session: Any) -> list[str | None]:
    return [
        m.get("reasoning_content")
        for m in session.payload()
        if m["role"] == "assistant"
    ]


def test_reasoning_is_not_sent_unless_the_model_needs_it(make_session: Any) -> None:
    session = make_session(tools=web_toolbox())
    session.messages = _two_turns()
    assert _sent_reasoning(session) == [None, None, None]


def test_turn_replay_sends_only_this_turns_reasoning(make_session: Any) -> None:
    session = _replaying(make_session(tools=web_toolbox()), "turn")
    session.messages = _two_turns()
    assert _sent_reasoning(session) == [None, None, "plan B"]


def test_all_replay_sends_every_assistant_messages_reasoning(make_session: Any) -> None:
    session = _replaying(make_session(tools=web_toolbox()), "all")
    session.messages = _two_turns()
    assert _sent_reasoning(session) == ["plan A", "wrap A", "plan B"]


def test_a_tool_loop_sends_the_rounds_reasoning_back(make_session: Any) -> None:
    thinking = FakeChunk([FakeChoice(FakeDelta(reasoning_content="search first"))])
    rounds = [
        [thinking, *tool_call_chunks("web_search", '{"query": "x"}')],
        [text_chunk("done")],
    ]
    stub = ToolBox([Tool("web_search", "s", {"type": "object"}, lambda **_: "r")])
    session = _replaying(make_session(rounds=rounds, tools=stub), "turn")
    list(session.stream("hi"))
    second = calls_of(session)[1]["messages"]
    asked = next(m for m in second if m.get("tool_calls"))
    assert asked["reasoning_content"] == "search first"
