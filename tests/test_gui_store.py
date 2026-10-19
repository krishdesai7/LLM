"""Chats on disk: append-only JSONL, tolerant of damage."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from llm.chat import Message, ToolCall
from llm.gui.store import ChatNotFoundError, ChatStore, Settings, default_root

if TYPE_CHECKING:
    from pathlib import Path

SETTINGS = Settings(model="fake/fake-1")


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    return ChatStore(tmp_path / "chats")


def _call(call_id: str) -> ToolCall:
    return ToolCall(call_id, "web_search", '{"query":"x"}')


def test_round_trip(store: ChatStore) -> None:
    chat_id = store.create(Settings("fake/fake-1", "be brief", 0.5, True))
    sent = [
        Message("user", "hi"),
        Message("assistant", "", [_call("c1")], reasoning="hmm", reasoning_seconds=2.5),
        Message("tool", "result", tool_call_id="c1"),
        Message("assistant", "done", reasoning="again", reasoning_seconds=1.0),
    ]
    for message in sent:
        store.append_message(chat_id, message)
    loaded = store.load(chat_id)
    assert loaded.messages == sent
    assert loaded.settings == Settings("fake/fake-1", "be brief", 0.5, True)
    assert loaded.id == chat_id


def test_latest_settings_line_wins(store: ChatStore) -> None:
    chat_id = store.create(SETTINGS)
    store.set_settings(chat_id, Settings("fake/other", None, None, True))
    assert store.load(chat_id).settings == Settings("fake/other", None, None, True)


def test_settings_lines_are_appended_not_rewritten(
    store: ChatStore, tmp_path: Path
) -> None:
    chat_id = store.create(SETTINGS)
    store.append_message(chat_id, Message("user", "hi"))
    path = tmp_path / "chats" / f"{chat_id}.jsonl"
    before = path.read_bytes()
    store.set_settings(chat_id, Settings("fake/other"))
    assert path.read_bytes().startswith(before)


def test_truncated_last_line_ignored(store: ChatStore, tmp_path: Path) -> None:
    chat_id = store.create(SETTINGS)
    store.append_message(chat_id, Message("user", "hi"))
    path = tmp_path / "chats" / f"{chat_id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "message", "role": "assis')
    assert [m.content for m in store.load(chat_id).messages] == ["hi"]


def test_dangling_tool_calls_dropped_on_load(store: ChatStore) -> None:
    chat_id = store.create(SETTINGS)
    store.append_message(chat_id, Message("user", "hi"))
    store.append_message(chat_id, Message("assistant", "", [_call("c1")]))
    assert [m.role for m in store.load(chat_id).messages] == ["user"]


def test_complete_tool_run_is_kept(store: ChatStore) -> None:
    chat_id = store.create(SETTINGS)
    for message in (
        Message("user", "hi"),
        Message("assistant", "", [_call("c1")]),
        Message("tool", "r", tool_call_id="c1"),
        Message("assistant", "answer"),
    ):
        store.append_message(chat_id, message)
    assert len(store.load(chat_id).messages) == 4


def test_partial_tool_run_drops_the_assistant_and_its_results(
    store: ChatStore,
) -> None:
    chat_id = store.create(SETTINGS)
    for message in (
        Message("user", "hi"),
        Message("assistant", "", [_call("c1"), _call("c2")]),
        Message("tool", "r", tool_call_id="c1"),
    ):
        store.append_message(chat_id, message)
    assert [m.role for m in store.load(chat_id).messages] == ["user"]


def test_summaries_newest_first_and_title_trimmed(
    store: ChatStore, tmp_path: Path
) -> None:
    older = store.create(SETTINGS)
    store.append_message(older, Message("user", "   hello\n\nworld  "))
    newer = store.create(SETTINGS)
    store.append_message(newer, Message("user", "x" * 100))
    root = tmp_path / "chats"
    os.utime(root / f"{older}.jsonl", (1_000, 1_000))
    os.utime(root / f"{newer}.jsonl", (2_000, 2_000))
    summaries = store.summaries()
    assert [s.id for s in summaries] == [newer, older]
    assert summaries[1].title == "hello world"
    assert summaries[0].title == "x" * 60 + "…"
    assert summaries[0].model == "fake/fake-1"


def test_summaries_skip_corrupt_and_foreign_files(
    store: ChatStore, tmp_path: Path
) -> None:
    good = store.create(SETTINGS)
    root = tmp_path / "chats"
    (root / "empty.jsonl").write_text("")
    (root / "garbage.jsonl").write_bytes(b"\xff\xfe\x00\x01")
    (root / "notes.txt").write_text("not a chat")
    (root / "headless.jsonl").write_text(
        '{"type": "message", "role": "user", "content": "x"}\n'
    )
    assert [s.id for s in store.summaries()] == [good]


@pytest.mark.parametrize("bad", ["../x", "a/b", "..\\x", "", "x.jsonl", "..", "a b"])
def test_ids_cannot_escape_root(store: ChatStore, bad: str) -> None:
    with pytest.raises(ChatNotFoundError):
        store.load(bad)
    with pytest.raises(ChatNotFoundError):
        store.append_message(bad, Message("user", "hi"))
    with pytest.raises(ChatNotFoundError):
        store.set_settings(bad, SETTINGS)
    with pytest.raises(ChatNotFoundError):
        store.delete(bad)


def test_delete_removes_file(store: ChatStore) -> None:
    chat_id = store.create(SETTINGS)
    store.delete(chat_id)
    assert store.summaries() == []
    with pytest.raises(ChatNotFoundError):
        store.load(chat_id)


def test_load_missing_raises_chat_not_found(store: ChatStore) -> None:
    with pytest.raises(ChatNotFoundError):
        store.load("20261018T101530-abcdef")


def test_default_root_honours_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_root() == tmp_path / "llm" / "chats"
