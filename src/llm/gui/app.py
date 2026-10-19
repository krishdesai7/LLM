"""The HTTP API behind the page: chats, settings, and streamed replies.

Every route sits behind :class:`~llm.gui.security.Guard`. Replies are started
here but live in :class:`~llm.gui.runs.RunManager`, so the request that carries
a reply's events is only a window onto it: closing or leaving it cancels nothing.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from litestar import Litestar, MediaType, Request, Response, delete, get, post, put
from litestar.datastructures import State
from litestar.exceptions import ClientException, HTTPException
from litestar.middleware import DefineMiddleware
from litestar.response import ServerSentEvent
from litestar.static_files import create_static_files_router

from .. import catalog, ui
from ..registry import LLMError, Registry, UnknownModelError
from .runs import ChatRun, RunBusyError, RunManager
from .security import Guard
from .sessions import apply_settings, build_session
from .store import ChatNotFoundError, ChatStore, Settings, StoredChat, title_of

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from ..chat import ChatSession, Message

__all__ = ["create_app"]

STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class SettingsBody:
    model: str | None = None
    system: str | None = None
    temperature: float | None = None
    web: bool = False


@dataclass
class ReplyBody:
    prompt: str
    owner: str


@dataclass
class AttachBody:
    after: int = 0


@dataclass
class OwnerBody:
    owner: str


class Hub:
    """Shared state: the registry, the store, and one live session per chat."""

    def __init__(
        self,
        registry: Registry,
        store: ChatStore,
        runs: RunManager,
        reap_interval: float,
    ) -> None:
        self.registry = registry
        self.store = store
        self.runs = runs
        self.reap_interval = reap_interval
        self._sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()

    def session(self, chat_id: str) -> ChatSession:
        """The chat's live session, resuming it from disk on first use."""
        with self._lock:
            cached = self._sessions.get(chat_id)
        if cached is not None:
            return cached
        built = _llm(lambda: build_session(self.registry, self.store.load(chat_id)))
        with self._lock:
            return self._sessions.setdefault(chat_id, built)

    def reconfigure(self, stored: StoredChat, settings: Settings) -> None:
        """Apply ``settings`` to the chat's session, or raise 400 and change nothing."""
        with self._lock:
            session = self._sessions.get(stored.id)
        if session is not None:
            _llm(lambda: apply_settings(session, self.registry, settings))
            return
        built = _llm(
            lambda: build_session(self.registry, replace(stored, settings=settings))
        )
        with self._lock:
            self._sessions.setdefault(stored.id, built)

    def forget(self, chat_id: str) -> None:
        with self._lock:
            self._sessions.pop(chat_id, None)


def create_app(
    registry: Registry,
    store: ChatStore,
    token: str,
    port: int,
    *,
    runs: RunManager | None = None,
    reap_interval: float = 5.0,
) -> Litestar:
    """The Litestar app, guarded by ``token`` and bound to ``port`` on loopback."""
    manager = runs if runs is not None else RunManager()
    hub = Hub(registry, store, manager, reap_interval)
    return Litestar(
        [
            *_ROUTES,
            create_static_files_router(path="/static", directories=[STATIC_DIR]),
        ],
        middleware=[DefineMiddleware(Guard, token=token, port=port)],
        exception_handlers={ChatNotFoundError: _chat_not_found},
        lifespan=[_reaper],
        state=State({"hub": hub}),
    )


# -- pages and metadata -----------------------------------------------------------


@get("/", sync_to_thread=True)
def index() -> Response[bytes]:
    page = (STATIC_DIR / "index.html").read_bytes()
    return Response(page, media_type=MediaType.HTML)


@get("/api/state", sync_to_thread=True)
def app_state(state: State) -> dict[str, Any]:
    hub: Hub = state.hub
    return {
        "backend": ui.search_status(),
        "providers": [
            {"name": p.name, "title": p.title, "ready": p.is_ready()}
            for p in hub.registry.sorted_providers()
        ],
        "default_model": hub.registry.default_model,
    }


@get("/api/models/{provider:str}", sync_to_thread=True)
def models(provider: str, state: State) -> list[str]:
    hub: Hub = state.hub
    try:
        target = hub.registry.provider(provider)
    except UnknownModelError:
        raise HTTPException(status_code=404, detail="unknown provider") from None
    return catalog.list_models(target)


# -- chats ------------------------------------------------------------------------


@get("/api/chats", sync_to_thread=True)
def list_chats(state: State) -> list[dict[str, Any]]:
    hub: Hub = state.hub
    running = hub.runs.running_ids()
    return [
        {**asdict(summary), "running": summary.id in running}
        for summary in hub.store.summaries()
    ]


@post("/api/chats", sync_to_thread=True)
def create_chat(data: SettingsBody, state: State) -> dict[str, str]:
    hub: Hub = state.hub
    settings = _settings_of(data, hub.registry)
    _llm(lambda: hub.registry.open(settings.model))  # fail now, not on first reply
    return {"id": hub.store.create(settings)}


@get("/api/chats/{chat_id:str}", sync_to_thread=True)
def get_chat(chat_id: str, state: State) -> dict[str, Any]:
    hub: Hub = state.hub
    stored = hub.store.load(chat_id)
    return {
        "id": stored.id,
        "title": title_of(stored.messages),
        "settings": asdict(stored.settings),
        "messages": [_message_json(m) for m in stored.messages],
        "running": hub.runs.get(chat_id) is not None,
    }


@delete("/api/chats/{chat_id:str}", status_code=204, sync_to_thread=True)
def delete_chat(chat_id: str, state: State) -> None:
    hub: Hub = state.hub
    hub.store.load(chat_id)  # 404 for anything that is not a chat
    if (run := hub.runs.get(chat_id)) is not None:
        run.cancel()
    hub.forget(chat_id)
    hub.store.delete(chat_id)


@put("/api/chats/{chat_id:str}/settings", status_code=204, sync_to_thread=True)
def put_settings(chat_id: str, data: SettingsBody, state: State) -> None:
    hub: Hub = state.hub
    stored = hub.store.load(chat_id)
    if hub.runs.get(chat_id) is not None:
        raise HTTPException(status_code=409, detail="a reply is in progress")
    settings = _settings_of(data, hub.registry)
    hub.reconfigure(stored, settings)
    hub.store.set_settings(chat_id, settings)


# -- replies ----------------------------------------------------------------------


@post("/api/chats/{chat_id:str}/reply", status_code=200, sync_to_thread=True)
def reply(chat_id: str, data: ReplyBody, state: State) -> ServerSentEvent:
    hub: Hub = state.hub
    hub.store.load(chat_id)  # 404 first
    prompt = data.prompt.strip()
    if not prompt:
        raise ClientException(detail="the prompt is empty")
    session = hub.session(chat_id)
    try:
        run = hub.runs.start(chat_id, session, hub.store, prompt, data.owner)
    except RunBusyError:
        raise HTTPException(
            status_code=409, detail="a reply is already in progress"
        ) from None
    return ServerSentEvent(_sse(run, 0))


@post("/api/chats/{chat_id:str}/attach", status_code=200, sync_to_thread=True)
def attach(chat_id: str, data: AttachBody, state: State) -> ServerSentEvent:
    hub: Hub = state.hub
    run = hub.runs.get(chat_id)
    if run is None:
        raise HTTPException(status_code=404, detail="no reply in progress")
    return ServerSentEvent(_sse(run, data.after))


@post("/api/chats/{chat_id:str}/stop", status_code=204, sync_to_thread=True)
def stop(chat_id: str, state: State) -> None:
    hub: Hub = state.hub
    if (run := hub.runs.get(chat_id)) is not None:
        run.cancel()


@post("/api/heartbeat", status_code=204, sync_to_thread=False)
def heartbeat(data: OwnerBody, state: State) -> None:
    hub: Hub = state.hub
    hub.runs.heartbeat(data.owner)
    hub.runs.reap()


_ROUTES = [
    index,
    app_state,
    models,
    list_chats,
    create_chat,
    get_chat,
    delete_chat,
    put_settings,
    reply,
    attach,
    stop,
    heartbeat,
]


# -- helpers ----------------------------------------------------------------------


@asynccontextmanager
async def _reaper(app: Litestar) -> AsyncGenerator[None]:
    """Cancel replies whose page vanished without a word."""
    hub: Hub = app.state.hub

    async def sweep() -> None:
        while True:
            await asyncio.sleep(hub.reap_interval)
            hub.runs.reap()

    task = asyncio.create_task(sweep())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


def _chat_not_found(
    _request: Request[Any, Any, Any], _exc: Exception
) -> Response[dict[str, str]]:
    return Response({"detail": "no such chat"}, status_code=404)


def _llm[T](call: Callable[[], T]) -> T:
    """Run ``call``, turning provider/registry failures into a 400."""
    try:
        return call()
    except LLMError as exc:
        raise ClientException(detail=str(exc)) from exc


def _settings_of(data: SettingsBody, registry: Registry) -> Settings:
    model = data.model or registry.default_model
    if not model:
        raise ClientException(detail="no model given and no default_model is set")
    return Settings(model, data.system, data.temperature, data.web)


async def _sse(run: ChatRun, after: int) -> AsyncGenerator[str]:
    async for event in run.subscribe(after):
        yield json.dumps(event)


def _message_json(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "content": message.content,
        "tool_calls": [asdict(call) for call in message.tool_calls],
        "tool_call_id": message.tool_call_id,
        "reasoning": message.reasoning,
        "reasoning_seconds": message.reasoning_seconds,
    }
