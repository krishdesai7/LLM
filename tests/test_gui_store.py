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


def test_a_dangling_tool_run_in_the_middle_keeps_later_messages(
    store: ChatStore,
) -> None:
    """After a crash, the user resumes and keeps chatting; that must survive."""
    chat_id = store.create(SETTINGS)
    for message in (
        Message("user", "first"),
        Message("assistant", "", [_call("c1"), _call("c2")]),
        Message("tool", "r", tool_call_id="c1"),
        Message("user", "second"),
        Message("assistant", "reply"),
    ):
        store.append_message(chat_id, message)
    assert [(m.role, m.content) for m in store.load(chat_id).messages] == [
        ("user", "first"),
        ("user", "second"),
        ("assistant", "reply"),
    ]


def test_an_append_after_a_truncated_last_line_is_not_swallowed(
    store: ChatStore, tmp_path: Path
) -> None:
    """A crash mid-write leaves no newline; the next message must still load."""
    chat_id = store.create(SETTINGS)
    store.append_message(chat_id, Message("user", "first"))
    path = tmp_path / "chats" / f"{chat_id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "message", "role": "assis')
    store.append_message(chat_id, Message("user", "after the crash"))
    assert [m.content for m in store.load(chat_id).messages] == [
        "first",
        "after the crash",
    ]


@pytest.mark.parametrize(
    "line",
    [
        '{"type": "message", "role": "user", "content": 5}',
        '{"type": "message", "role": "user", "content": ["x"]}',
        '{"type": "message", "role": "assistant", "content": "x", "reasoning": 3}',
        '{"type": "message", "role": "assistant", "reasoning_seconds": "a"}',
        '{"type": "message", "role": "tool", "content": "x", "tool_call_id": 7}',
        '{"type": "message", "role": "robot", "content": "x"}',
        '{"type": "message", "role": "assistant", "tool_calls": [{"id": 1}]}',
        '{"type": "message", "role": "assistant", "tool_calls": "nope"}',
        '{"type": "settings", "model": 5}',
        '{"type": "settings", "model": "fake/fake-1", "temperature": "hot"}',
        "[1, 2, 3]",
        '"just a string"',
        "null",
    ],
)
def test_wrongly_typed_lines_are_skipped_not_fatal(
    store: ChatStore, tmp_path: Path, line: str
) -> None:
    good = store.create(SETTINGS)
    store.append_message(good, Message("user", "fine"))
    path = tmp_path / "chats" / f"{good}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    assert [m.content for m in store.load(good).messages] == ["fine"]
    assert [s.id for s in store.summaries()] == [good]
    assert store.load(good).settings == SETTINGS


def test_a_null_content_message_does_not_break_listing(
    store: ChatStore, tmp_path: Path
) -> None:
    chat_id = store.create(SETTINGS)
    path = tmp_path / "chats" / f"{chat_id}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"type": "message", "role": "user", "content": null}\n')
    [summary] = store.summaries()
    assert summary.title == "New chat"


@pytest.mark.parametrize("created", ["null", '"yesterday"', "[1]"])
def test_a_wrongly_typed_created_time_falls_back_to_the_file_time(
    store: ChatStore, tmp_path: Path, created: str
) -> None:
    root = tmp_path / "chats"
    root.mkdir()
    (root / "odd-meta.jsonl").write_text(
        f'{{"type": "meta", "id": "odd-meta", "created": {created}}}\n'
        '{"type": "settings", "model": "fake/fake-1"}\n'
        '{"type": "message", "role": "user", "content": "hi"}\n'
    )
    assert [s.id for s in store.summaries()] == ["odd-meta"]
    assert store.load("odd-meta").created > 0


def test_lone_surrogates_in_model_text_do_not_break_appends(store: ChatStore) -> None:
    """Providers can split an escaped emoji pair across chunks."""
    chat_id = store.create(SETTINGS)
    store.append_message(chat_id, Message("assistant", "a\ud83db"))
    store.append_message(chat_id, Message("user", "still works"))
    assert [m.content for m in store.load(chat_id).messages] == [
        "a?b",
        "still works",
    ]


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The chat files read from disk, in order."""
    from llm.gui import store as store_module

    seen: list[str] = []
    real = store_module._parse

    def counting(path: Path) -> object:
        seen.append(path.stem)
        return real(path)

    monkeypatch.setattr(store_module, "_parse", counting)
    return seen


def test_unchanged_chats_are_not_reread_for_the_sidebar(
    store: ChatStore, parses: list[str]
) -> None:
    chat_id = store.create(SETTINGS)
    store.append_message(chat_id, Message("user", "hello"))
    first = store.summaries()
    parses.clear()
    assert store.summaries() == first
    assert parses == []


def test_only_the_changed_chat_is_reread(store: ChatStore, parses: list[str]) -> None:
    quiet = store.create(SETTINGS)
    busy = store.create(SETTINGS)
    store.summaries()
    store.append_message(busy, Message("user", "now with a title"))
    parses.clear()
    titles = {s.id: s.title for s in store.summaries()}
    assert parses == [busy]
    assert titles[busy] == "now with a title"
    assert titles[quiet] == "New chat"


def test_unusable_files_are_not_reread_either(
    store: ChatStore, tmp_path: Path, parses: list[str]
) -> None:
    good = store.create(SETTINGS)
    (tmp_path / "chats" / "garbage.jsonl").write_bytes(b"\xff\xfe\x00\x01")
    store.summaries()
    parses.clear()
    assert [s.id for s in store.summaries()] == [good]
    assert parses == []


def test_a_deleted_chat_leaves_the_sidebar_and_a_new_one_joins(
    store: ChatStore,
) -> None:
    gone = store.create(SETTINGS)
    store.summaries()
    store.delete(gone)
    fresh = store.create(SETTINGS)
    assert [s.id for s in store.summaries()] == [fresh]
