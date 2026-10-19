"""Replies as server-side tasks, independent of the browser that asked.

A :class:`ChatRun` drives ``ChatSession.stream`` on a worker thread, buffers
every event, and persists each message as soon as it exists. Subscribers
replay the buffer and then follow it live, so a page can leave (switch chats)
and come back mid-reply without losing anything. :class:`RunManager` keeps one
unfinished run per chat and cancels runs whose page stopped sending heartbeats.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import closing
from typing import TYPE_CHECKING, Any

from openai import OpenAIError

from ..chat import Message
from ..registry import LLMError
from .store import ChatNotFoundError

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from ..chat import ChatSession
    from .store import ChatStore

__all__ = ["ChatRun", "RunBusyError", "RunManager"]

#: How long a page may go without a heartbeat before its replies are cancelled.
GRACE_SECONDS = 15.0


class RunBusyError(Exception):
    """The chat already has a reply in progress."""


class ChatRun:
    """One reply, from the user's prompt to the final ``done`` event.

    Events are dicts with a ``seq`` (their index in :attr:`events`) and a
    ``type``: ``text``, ``reasoning``, ``tool_call``, ``tool_result`` or
    ``notice`` (each with ``text``), then optionally ``usage`` and ``error``,
    and always a last ``done``.
    """

    def __init__(
        self,
        chat_id: str,
        session: ChatSession,
        store: ChatStore,
        prompt: str,
        owner: str,
    ) -> None:
        self.chat_id = chat_id
        self.session = session
        self.store = store
        self.prompt = prompt
        self.owner = owner
        self.events: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Event]] = []
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._thread = threading.Thread(
            target=self._work, name=f"reply-{chat_id}", daemon=True
        )

    @property
    def finished(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        """Stop after the next streamed fragment; what arrived is kept."""
        self._cancel.set()

    async def subscribe(self, after: int = 0) -> AsyncGenerator[dict[str, Any]]:
        """Yield events from index ``after`` on, ending after ``done``."""
        loop = asyncio.get_running_loop()
        wake = asyncio.Event()
        entry = (loop, wake)
        with self._lock:
            self._waiters.append(entry)
        index = max(after, 0)
        try:
            while True:
                wake.clear()
                with self._lock:
                    fresh = self.events[index:]
                    ended = bool(self.events) and self.events[-1]["type"] == "done"
                for event in fresh:
                    yield event
                    index += 1
                    if event["type"] == "done":
                        return
                if not fresh:
                    if ended:
                        return
                    await wake.wait()
        finally:
            with self._lock:
                self._waiters.remove(entry)

    # -- worker thread --------------------------------------------------------

    def _work(self) -> None:
        session = self.session
        persisted = len(session.messages) + 1  # the user message, saved below
        try:
            self.store.append_message(self.chat_id, Message("user", self.prompt))
            with closing(session.stream(self.prompt)) as stream:
                for delta in stream:
                    self._emit({"type": delta.kind, "text": delta.text})
                    if delta.kind not in ("text", "reasoning"):
                        persisted = self._flush(persisted)
                    if self._cancel.is_set():
                        break
        except (OpenAIError, LLMError) as exc:
            self._emit({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
        # ruff: ignore[blind-except] - a worker must always end with `done`
        except Exception as exc:
            self._emit({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
        finally:
            self._finish(persisted)

    def _finish(self, persisted: int) -> None:
        try:
            self._flush(persisted)
        except (OSError, ChatNotFoundError) as exc:
            self._emit({"type": "error", "text": f"could not save chat: {exc}"})
        usage = self.session.last_usage
        if usage is not None and usage.total:
            self._emit(
                {
                    "type": "usage",
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                }
            )
        self._emit({"type": "done", "cancelled": self._cancel.is_set()})
        self._done.set()

    def _flush(self, persisted: int) -> int:
        """Append any session messages not yet on disk; return the new count."""
        pending = self.session.messages[persisted:]
        for message in pending:
            self.store.append_message(self.chat_id, message)
        return persisted + len(pending)

    def _emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            stamped = {"seq": len(self.events), **event}
            self.events.append(stamped)
            waiters = list(self._waiters)
        for loop, wake in waiters:
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:  # that subscriber's loop has already closed
                continue


class RunManager:
    """The unfinished runs, one per chat, plus who is still watching them."""

    def __init__(
        self,
        grace_seconds: float = GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._grace = grace_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._runs: dict[str, ChatRun] = {}
        self._beats: dict[str, float] = {}

    def start(
        self,
        chat_id: str,
        session: ChatSession,
        store: ChatStore,
        prompt: str,
        owner: str,
    ) -> ChatRun:
        with self._lock:
            current = self._runs.get(chat_id)
            if current is not None and not current.finished:
                raise RunBusyError(chat_id)
            run = ChatRun(chat_id, session, store, prompt, owner)
            self._runs[chat_id] = run
            self._beats[owner] = self._clock()
        run.start()
        return run

    def get(self, chat_id: str) -> ChatRun | None:
        """The chat's reply in progress, if any."""
        with self._lock:
            run = self._runs.get(chat_id)
        return run if run is not None and not run.finished else None

    def running_ids(self) -> set[str]:
        with self._lock:
            return {cid for cid, run in self._runs.items() if not run.finished}

    def heartbeat(self, owner: str) -> None:
        with self._lock:
            self._beats[owner] = self._clock()

    def reap(self) -> None:
        """Cancel replies whose page has not been heard from within the grace."""
        now = self._clock()
        with self._lock:
            runs = [run for run in self._runs.values() if not run.finished]
            beats = dict(self._beats)
        for run in runs:
            if now - beats.get(run.owner, 0.0) > self._grace:
                run.cancel()
