"""Live ``ChatSession`` objects from stored chats and settings changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import tools
from ..chat import ChatSession

if TYPE_CHECKING:
    from ..registry import Registry
    from .store import Settings, StoredChat

__all__ = ["apply_settings", "build_session"]


def build_session(registry: Registry, stored: StoredChat) -> ChatSession:
    """Resume ``stored`` on the model and settings it last used.

    Registry errors (unknown model, missing API key) propagate unchanged.
    """
    settings = stored.settings
    ref, client = registry.open(settings.model)
    return ChatSession(
        ref=ref,
        client=client,
        system=settings.system,
        messages=list(stored.messages),
        params=_params(settings),
        tools=tools.web_toolbox() if settings.web else None,
    )


def apply_settings(
    session: ChatSession, registry: Registry, settings: Settings
) -> None:
    """Make ``session`` match ``settings``, the way the REPL's commands do.

    The model is opened first, so a bad one raises before anything changes.
    """
    if settings.model != session.ref.spec:
        ref, client = registry.open(settings.model)
        session.switch(ref, client)
    session.system = settings.system
    session.params = _params(settings)
    session.tools = tools.web_toolbox() if settings.web else None


def _params(settings: Settings) -> dict[str, float]:
    return {} if settings.temperature is None else {"temperature": settings.temperature}
