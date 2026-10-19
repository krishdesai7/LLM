"""Replies run on worker threads, buffered, persisted, and cancellable."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING, Any

import openai
import pytest
from conftest import (
    REF,
    FakeClient,
    GatedClient,
    default_round,
    text_chunk,
    tool_call_chunks,
)

from llm.chat import ChatSession, Message
from llm.gui.runs import ChatRun, RunBusyError, RunManager
from llm.gui.store import ChatStore, Settings
from llm.tools import Tool, ToolBox

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class BrokenClient:
    def __init__(self) -> None:
        def create(**_kwargs: Any) -> Any:
            msg = "connection dropped"
            raise openai.OpenAIError(msg)

        completions = type("_Completions", (), {"create": staticmethod(create)})()
        self.chat = type("_Chat", (), {"completions": completions})()


def _session(client: Any, **kwargs: Any) -> ChatSession:
    return ChatSession(ref=REF, client=client, **kwargs)


def _echo_box() -> ToolBox:
    return ToolBox(
        [
            Tool(
                name="web_search",
                description="search",
                parameters={"type": "object", "properties": {}},
                run=lambda **kw: f"echo {kw}",
            )
        ]
    )


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    return ChatStore(tmp_path / "chats")


@pytest.fixture
def chat_id(store: ChatStore) -> str:
    return store.create(Settings("fake/fake-1"))


def _collect(run: ChatRun, after: int = 0) -> list[dict[str, Any]]:
    async def gather() -> list[dict[str, Any]]:
        return [event async for event in run.subscribe(after)]

    return asyncio.run(asyncio.wait_for(gather(), timeout=10))


def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _start(
    chat_id: str, store: ChatStore, session: ChatSession, prompt: str = "hi"
) -> ChatRun:
    run = ChatRun(chat_id, session, store, prompt, owner="page-1")
    run.start()
    return run


def test_events_mirror_deltas_then_usage_then_done(
    chat_id: str, store: ChatStore
) -> None:
    run = _start(chat_id, store, _session(FakeClient([default_round()])))
    events = _collect(run)
    assert [(e["type"], e.get("text")) for e in events[:3]] == [
        ("reasoning", "hmm"),
        ("text", "Hello"),
        ("text", ", world"),
    ]
    assert events[3] == {"seq": 3, "type": "usage", "prompt": 7, "completion": 3}
    assert events[4] == {"seq": 4, "type": "done", "cancelled": False}
    assert [e["seq"] for e in events] == list(range(5))


def test_messages_are_persisted_before_done(chat_id: str, store: ChatStore) -> None:
    run = _start(chat_id, store, _session(FakeClient([default_round()])))
    _collect(run)
    saved = store.load(chat_id).messages
    assert [(m.role, m.content, m.reasoning) for m in saved] == [
        ("user", "hi", ""),
        ("assistant", "Hello, world", "hmm"),
    ]
    assert run.finished


def test_user_message_is_persisted_before_the_first_event(
    chat_id: str, store: ChatStore
) -> None:
    client = GatedClient([default_round()], gates=[(0, 0)])
    run = _start(chat_id, store, _session(client))
    assert client.reached[0, 0].wait(5)
    assert [m.content for m in store.load(chat_id).messages] == ["hi"]
    assert run.events == []
    client.gates[0, 0].set()
    _collect(run)


def test_each_tool_round_is_persisted_before_the_next_starts(
    chat_id: str, store: ChatStore
) -> None:
    rounds = [tool_call_chunks("web_search", "{}"), [text_chunk("done")]]
    client = GatedClient(rounds, gates=[(1, 0)])
    run = _start(chat_id, store, _session(client, tools=_echo_box()))
    assert client.reached[1, 0].wait(5)
    assert [m.role for m in store.load(chat_id).messages] == [
        "user",
        "assistant",
        "tool",
    ]
    client.gates[1, 0].set()
    _collect(run)
    assert store.load(chat_id).messages[-1].content == "done"


def test_cancel_saves_the_partial_reply_and_reports_cancelled(
    chat_id: str, store: ChatStore
) -> None:
    rounds = [[text_chunk("Hello"), text_chunk(", world"), text_chunk("!")]]
    client = GatedClient(rounds, gates=[(0, 1)])
    run = _start(chat_id, store, _session(client))
    assert client.reached[0, 1].wait(5)
    run.cancel()
    client.gates[0, 1].set()
    events = _collect(run)
    assert events[-1] == {"seq": len(events) - 1, "type": "done", "cancelled": True}
    saved = store.load(chat_id).messages
    assert saved[-1].role == "assistant"
    assert saved[-1].content.startswith("Hello")
    assert not saved[-1].content.endswith("!")


def test_subscribe_after_replays_only_newer_events(
    chat_id: str, store: ChatStore
) -> None:
    run = _start(chat_id, store, _session(FakeClient([default_round()])))
    full = _collect(run)
    assert _collect(run, after=3) == full[3:]


def test_a_late_subscriber_gets_the_whole_reply(chat_id: str, store: ChatStore) -> None:
    run = _start(chat_id, store, _session(FakeClient([default_round()])))
    _wait(lambda: run.finished)
    assert [e["type"] for e in _collect(run)][-1] == "done"
    assert len(_collect(run)) == 5


def test_two_subscribers_see_the_same_events(chat_id: str, store: ChatStore) -> None:
    client = GatedClient([default_round()], gates=[(0, 1)])
    run = _start(chat_id, store, _session(client))
    results: list[list[dict[str, Any]]] = []
    threads = [
        threading.Thread(target=lambda: results.append(_collect(run))) for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    assert client.reached[0, 1].wait(5)
    client.gates[0, 1].set()
    for thread in threads:
        thread.join(10)
    assert len(results) == 2
    assert results[0] == results[1]


def test_a_second_run_for_the_same_chat_is_refused(
    chat_id: str, store: ChatStore
) -> None:
    client = GatedClient([default_round()], gates=[(0, 0)])
    runs = RunManager()
    first = runs.start(chat_id, _session(client), store, "hi", "page-1")
    with pytest.raises(RunBusyError):
        runs.start(chat_id, _session(client), store, "again", "page-1")
    assert runs.running_ids() == {chat_id}
    assert runs.get(chat_id) is first
    client.gates[0, 0].set()
    _collect(first)
    assert runs.running_ids() == set()
    assert runs.get(chat_id) is None


def test_different_chats_run_at_the_same_time(store: ChatStore) -> None:
    one, two = (
        store.create(Settings("fake/fake-1")),
        store.create(Settings("fake/fake-1")),
    )
    client_a = GatedClient([default_round()], gates=[(0, 0)])
    client_b = GatedClient([default_round()], gates=[(0, 0)])
    runs = RunManager()
    run_a = runs.start(one, _session(client_a), store, "a", "p")
    run_b = runs.start(two, _session(client_b), store, "b", "p")
    assert client_a.reached[0, 0].wait(5)
    assert client_b.reached[0, 0].wait(5)
    assert runs.running_ids() == {one, two}
    client_a.gates[0, 0].set()
    client_b.gates[0, 0].set()
    _collect(run_a)
    _collect(run_b)


def test_a_provider_error_becomes_an_error_event_and_the_chat_stays_usable(
    chat_id: str, store: ChatStore
) -> None:
    session = _session(BrokenClient())
    failed = _start(chat_id, store, session)
    events = _collect(failed)
    assert [e["type"] for e in events] == ["error", "done"]
    assert "OpenAIError" in events[0]["text"]
    assert [m.content for m in store.load(chat_id).messages] == ["hi"]

    session.client = FakeClient([default_round()])  # type: ignore[assignment]
    retry = _start(chat_id, store, session, "again")
    assert _collect(retry)[-1]["type"] == "done"
    assert store.load(chat_id).messages[-1].content == "Hello, world"


def test_reap_cancels_runs_whose_page_has_gone(chat_id: str, store: ChatStore) -> None:
    now = [0.0]
    client = GatedClient([default_round()], gates=[(0, 0)])
    runs = RunManager(grace_seconds=15, clock=lambda: now[0])
    run = runs.start(chat_id, _session(client), store, "hi", "page-1")
    now[0] = 100.0
    runs.reap()
    assert run.cancelled
    client.gates[0, 0].set()
    assert _collect(run)[-1]["cancelled"] is True


def test_a_heartbeat_keeps_a_run_alive(chat_id: str, store: ChatStore) -> None:
    now = [0.0]
    client = GatedClient([default_round()], gates=[(0, 0)])
    runs = RunManager(grace_seconds=15, clock=lambda: now[0])
    run = runs.start(chat_id, _session(client), store, "hi", "page-1")
    now[0] = 10.0
    runs.heartbeat("page-1")
    now[0] = 20.0
    runs.reap()
    assert not run.cancelled
    client.gates[0, 0].set()
    _collect(run)


def test_events_are_json_serialisable(chat_id: str, store: ChatStore) -> None:
    run = _start(chat_id, store, _session(FakeClient([default_round()])))
    json.dumps(_collect(run))


def test_messages_of_other_chats_are_untouched(chat_id: str, store: ChatStore) -> None:
    other = store.create(Settings("fake/fake-1"))
    store.append_message(other, Message("user", "keep me"))
    _collect(_start(chat_id, store, _session(FakeClient([default_round()]))))
    assert [m.content for m in store.load(other).messages] == ["keep me"]


def test_abandoning_a_subscription_does_not_cancel_the_run(
    chat_id: str, store: ChatStore
) -> None:
    client = GatedClient([default_round()], gates=[(0, 1)])
    run = _start(chat_id, store, _session(client))

    async def read_one_then_leave() -> None:
        stream = run.subscribe()
        await anext(stream)
        await stream.aclose()

    asyncio.run(read_one_then_leave())
    client.gates[0, 1].set()
    _wait(lambda: run.finished)
    assert not run.cancelled
    assert store.load(chat_id).messages[-1].content == "Hello, world"


def test_deleting_the_chat_mid_run_still_ends_with_done(
    chat_id: str, store: ChatStore
) -> None:
    client = GatedClient([default_round()], gates=[(0, 1)])
    run = _start(chat_id, store, _session(client))
    assert client.reached[0, 1].wait(5)
    store.delete(chat_id)
    client.gates[0, 1].set()
    events = _collect(run)
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "error" for e in events)
