from __future__ import annotations

from typing import Any

import pytest
from conftest import FakeChunk, calls_of


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


def test_send_returns_the_reply_without_streaming(make_session: Any) -> None:
    session = make_session()
    assert session.send("hi") == "Hello, world"
    assert calls_of(session)[0]["stream"] is False


def test_system_prompt_leads_the_payload(make_session: Any) -> None:
    session = make_session(system="be terse")
    list(session.stream("hi"))
    sent = calls_of(session)[0]["messages"]
    assert sent[0] == {"role": "system", "content": "be terse"}
    assert sent[1] == {"role": "user", "content": "hi"}


def test_params_are_forwarded(make_session: Any) -> None:
    session = make_session(params={"temperature": 0.2})
    list(session.stream("hi"))
    assert calls_of(session)[0]["temperature"] == pytest.approx(0.2)


def test_multi_turn_context_is_accumulated(make_session: Any) -> None:
    session = make_session()
    list(session.stream("first"))
    list(session.stream("second"))
    sent = calls_of(session)[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]


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
