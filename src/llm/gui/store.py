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
        """Every readable chat, most recently updated first."""
        if not self.root.is_dir():
            return []
        found = [
            ChatSummary(
                stored.id,
                title_of(stored.messages),
                stored.updated,
                stored.settings.model,
            )
            for path in self.root.glob("*.jsonl")
            if _ID.fullmatch(path.stem) and (stored := _parse(path)) is not None
        ]
        return sorted(found, key=lambda summary: summary.updated, reverse=True)

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
        with path.open("a", encoding="utf-8") as handle:
            handle.write(_line(record))


def _line(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False) + "\n"


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


def _message_from_dict(record: dict[str, Any]) -> Message:
    return Message(
        role=record["role"],
        content=record.get("content", ""),
        tool_calls=[ToolCall(**call) for call in record.get("tool_calls") or []],
        tool_call_id=record.get("tool_call_id"),
        reasoning=record.get("reasoning", ""),
        reasoning_seconds=record.get("reasoning_seconds", 0.0),
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
                settings = Settings(
                    model=record["model"],
                    system=record.get("system"),
                    temperature=record.get("temperature"),
                    web=bool(record.get("web", False)),
                )
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
        created=float(meta.get("created", updated)),
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
