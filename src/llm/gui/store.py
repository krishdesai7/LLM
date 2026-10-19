"""Chats on disk: one append-only JSON Lines file per conversation.

Every line is an object with a ``type``: ``meta`` (first line), ``settings``
(written at creation and again whenever the user changes one; the latest
wins) or ``message``. Appending never rewrites earlier bytes, so a crash can
at worst truncate the last line, which loading ignores.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..chat import Message, ToolCall

__all__ = [
    "ChatNotFoundError",
    "ChatStore",
    "ChatSummary",
    "Settings",
    "StoredChat",
    "default_root",
    "title_of",
]

_ID = re.compile(r"[0-9A-Za-z_-]+")
_ROLES = ("system", "user", "assistant", "tool")
_TITLE_CHARS = 60


class ChatNotFoundError(LookupError):
    """No such chat, or the id is not one this store would ever issue."""


@dataclass(frozen=True, slots=True)
class Settings:
    """What the page can change about a chat after it starts."""

    model: str
    system: str | None = None
    temperature: float | None = None
    web: bool = False


@dataclass(frozen=True, slots=True)
class ChatSummary:
    id: str
    title: str
    updated: float
    model: str


@dataclass(slots=True)
class StoredChat:
    id: str
    created: float
    updated: float
    settings: Settings
    messages: list[Message]


def default_root() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local" / "share"
    return root / "llm" / "chats"


class ChatStore:
    """Reads and appends chat files under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        # path stem -> (mtime_ns, size) it was read at, and what it summarised to
        # (None: not a usable chat). Lets the sidebar skip files that didn't change.
        self._seen: dict[str, tuple[tuple[int, int], ChatSummary | None]] = {}

    def create(self, settings: Settings) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        chat_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"
        meta = {"type": "meta", "id": chat_id, "created": time.time()}
        with self._path(chat_id).open("x", encoding="utf-8") as handle:
            handle.write(_line(meta) + _line(_settings_to_dict(settings)))
        return chat_id

    def append_message(self, chat_id: str, message: Message) -> None:
        self._append(chat_id, _message_to_dict(message))

    def set_settings(self, chat_id: str, settings: Settings) -> None:
        self._append(chat_id, _settings_to_dict(settings))

    def load(self, chat_id: str) -> StoredChat:
        path = self._path(chat_id)
        stored = _parse(path) if path.is_file() else None
        if stored is None:
            raise ChatNotFoundError(chat_id)
        return stored

    def summaries(self) -> list[ChatSummary]:
        """Every readable chat, most recently updated first.

        Only files whose size or mtime changed since the last call are re-read.
        """
        if not self.root.is_dir():
            self._seen = {}
            return []
        seen: dict[str, tuple[tuple[int, int], ChatSummary | None]] = {}
        for path in self.root.glob("*.jsonl"):
            if _ID.fullmatch(path.stem) and (entry := self._entry(path)) is not None:
                seen[path.stem] = entry
        self._seen = seen
        found = [summary for _, summary in seen.values() if summary is not None]
        return sorted(found, key=lambda summary: summary.updated, reverse=True)

    def _entry(self, path: Path) -> tuple[tuple[int, int], ChatSummary | None] | None:
        try:
            info = path.stat()
        except OSError:
            return None
        stamp = (info.st_mtime_ns, info.st_size)
        known = self._seen.get(path.stem)
        if known is not None and known[0] == stamp:
            return known
        stored = _parse(path)
        summary = (
            None
            if stored is None
            else ChatSummary(
                stored.id,
                title_of(stored.messages),
                stored.updated,
                stored.settings.model,
            )
        )
        return stamp, summary

    def delete(self, chat_id: str) -> None:
        path = self._path(chat_id)
        try:
            path.unlink()
        except FileNotFoundError:
            raise ChatNotFoundError(chat_id) from None

    # -- internals ------------------------------------------------------------

    def _path(self, chat_id: str) -> Path:
        if not _ID.fullmatch(chat_id):
            raise ChatNotFoundError(chat_id)
        return self.root / f"{chat_id}.jsonl"

    def _append(self, chat_id: str, record: dict[str, Any]) -> None:
        path = self._path(chat_id)
        if not path.is_file():
            raise ChatNotFoundError(chat_id)
        # A crash mid-write leaves a line with no newline; start on a fresh line
        # rather than gluing this record onto the fragment.
        lead = "\n" if _ends_mid_line(path) else ""
        with path.open("a", encoding="utf-8") as handle:
            handle.write(lead + _line(record))


def _line(record: dict[str, Any]) -> str:
    # Lone surrogates (a split emoji pair from a provider) cannot be encoded;
    # they become "?" instead of failing the write.
    flat = json.dumps(record, ensure_ascii=False) + "\n"
    return flat.encode("utf-8", errors="replace").decode("utf-8")


def _ends_mid_line(path: Path) -> bool:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            return False
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) != b"\n"


def _settings_to_dict(settings: Settings) -> dict[str, Any]:
    return {
        "type": "settings",
        "model": settings.model,
        "system": settings.system,
        "temperature": settings.temperature,
        "web": settings.web,
    }


def _message_to_dict(message: Message) -> dict[str, Any]:
    return {
        "type": "message",
        "role": message.role,
        "content": message.content,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ],
        "tool_call_id": message.tool_call_id,
        "reasoning": message.reasoning,
        "reasoning_seconds": message.reasoning_seconds,
    }


def _text(value: Any) -> str:
    """A string field; ``null`` reads as empty, anything else is damage."""
    if value is None:
        return ""
    if not isinstance(value, str):
        msg = f"expected text, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _number(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"expected a number, got {type(value).__name__}"
        raise TypeError(msg)
    return float(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tool_call_from_dict(raw: Any) -> ToolCall:
    if not isinstance(raw, dict):
        msg = "a tool call must be an object"
        raise TypeError(msg)
    return ToolCall(
        id=_text(raw["id"]), name=_text(raw["name"]), arguments=_text(raw["arguments"])
    )


def _message_from_dict(record: dict[str, Any]) -> Message:
    role = record["role"]
    calls = record.get("tool_calls") or []
    if role not in _ROLES or not isinstance(calls, list):
        msg = "not a message"
        raise ValueError(msg)
    call_id = record.get("tool_call_id")
    return Message(
        role=role,
        content=_text(record.get("content")),
        tool_calls=[_tool_call_from_dict(call) for call in calls],
        tool_call_id=None if call_id is None else _text(call_id),
        reasoning=_text(record.get("reasoning")),
        reasoning_seconds=_number(record.get("reasoning_seconds"), 0.0),
    )


def _settings_from_dict(record: dict[str, Any]) -> Settings:
    model = record["model"]
    if not isinstance(model, str) or not model:
        msg = "a model name is required"
        raise ValueError(msg)
    system = record.get("system")
    temperature = record.get("temperature")
    return Settings(
        model=model,
        system=None if system is None else _text(system),
        temperature=None if temperature is None else _number(temperature, 0.0),
        web=bool(record.get("web")),
    )


def _parse(path: Path) -> StoredChat | None:
    """Read a chat file, or ``None`` if it is not a usable chat."""
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
        updated = path.stat().st_mtime
    except OSError:
        return None

    meta: dict[str, Any] | None = None
    settings: Settings | None = None
    messages: list[Message] = []
    for number, raw in enumerate(text.splitlines()):
        try:
            record = json.loads(raw)
            kind = record["type"]
            if number == 0:
                if kind != "meta":
                    return None
                meta = record
            elif kind == "settings":
                settings = _settings_from_dict(record)
            elif kind == "message":
                messages.append(_message_from_dict(record))
        except ValueError, KeyError, TypeError:
            if number == 0:
                return None
            continue  # a truncated or damaged line: skip it
    if meta is None or settings is None:
        return None
    return StoredChat(
        id=path.stem,
        created=_number(meta.get("created"), updated)
        if _is_number(meta.get("created"))
        else updated,
        updated=updated,
        settings=settings,
        messages=_without_dangling_tool_calls(messages),
    )


def _without_dangling_tool_calls(messages: list[Message]) -> list[Message]:
    """Drop every assistant tool request whose results are incomplete.

    The API rejects a ``tool_calls`` message without a reply to each call, which
    is what a stop or crash mid-tool-run leaves behind. Only that request and
    its partial results go; messages after it (the user resumed) are kept.
    """
    kept: list[Message] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        index += 1
        if not message.tool_calls:
            kept.append(message)
            continue
        results: list[Message] = []
        while index < len(messages) and messages[index].role == "tool":
            results.append(messages[index])
            index += 1
        answered = {result.tool_call_id for result in results}
        if {call.id for call in message.tool_calls} <= answered:
            kept.extend([message, *results])
    return kept


def title_of(messages: list[Message]) -> str:
    first = next((m.content for m in messages if m.role == "user"), "")
    flat = " ".join(first.split())
    if not flat:
        return "New chat"
    return flat if len(flat) <= _TITLE_CHARS else flat[:_TITLE_CHARS].rstrip() + "…"
