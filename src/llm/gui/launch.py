"""Start the GUI server and point a browser at it.

Granian runs the app in a worker process, so the launch settings (including the
per-launch token) travel in environment variables that the worker's
:func:`app_factory` reads; they are never written to disk or to the command line.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser
from dataclasses import replace
from typing import TYPE_CHECKING

from .. import ui
from ..registry import LLMError, Registry
from .app import create_app
from .security import make_token
from .store import ChatStore, default_root

if TYPE_CHECKING:
    from litestar import Litestar

__all__ = ["PortBusyError", "app_factory", "serve"]

# ruff: ignore[hardcoded-password-string] - the name of a variable, not a secret
ENV_TOKEN = "LLM_GUI_TOKEN"
ENV_PORT = "LLM_GUI_PORT"
ENV_MODEL = "LLM_GUI_MODEL"
ENV_KEYS = (ENV_TOKEN, ENV_PORT, ENV_MODEL)
HOST = "127.0.0.1"
READY_TIMEOUT = 15.0


class PortBusyError(LLMError):
    """Something is already listening on the requested port."""


def serve(
    registry: Registry, *, model: str | None, port: int, open_browser: bool
) -> None:
    """Run the GUI on ``127.0.0.1:port`` until interrupted."""
    if model:
        registry.resolve(model)  # fail on a typo now, not in the first chat
    _ensure_free(port)

    token = make_token()
    os.environ[ENV_TOKEN] = token
    os.environ[ENV_PORT] = str(port)
    if model:
        os.environ[ENV_MODEL] = model
    else:
        os.environ.pop(ENV_MODEL, None)

    url = f"http://{HOST}:{port}/?token={token}"
    ui.console.print(f"[meta]chat GUI at[/meta] {url}", soft_wrap=True, highlight=False)
    ui.console.print("[meta]ctrl-c to stop[/meta]")
    if open_browser:
        threading.Thread(target=_open_when_ready, args=(port, url), daemon=True).start()
    _run_server(port)


def app_factory() -> Litestar:
    """Build the app inside the server worker, from the launch environment."""
    registry = Registry.load()
    model = os.environ.get(ENV_MODEL)
    if model:
        registry = replace(registry, default_model=model)
    return create_app(
        registry,
        ChatStore(default_root()),
        os.environ[ENV_TOKEN],
        int(os.environ[ENV_PORT]),
    )


def _run_server(port: int) -> None:
    from granian import Granian
    from granian.constants import Interfaces

    Granian(
        "llm.gui.launch:app_factory",
        address=HOST,
        port=port,
        interface=Interfaces.ASGI,
        factory=True,
        log_access=False,
    ).serve()


def _ensure_free(port: int) -> None:
    with socket.socket() as probe:
        # As the server will: connections lingering from a previous run (TIME_WAIT)
        # must not make the port look taken.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((HOST, port))
        except OSError as exc:
            msg = f"port {port} is busy; pick another with --port"
            raise PortBusyError(msg) from exc


def _open_when_ready(port: int, url: str) -> None:
    """Wait for the server to accept connections, then open the browser."""
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        try:
            socket.create_connection((HOST, port), timeout=0.2).close()
        except OSError:
            time.sleep(0.1)
            continue
        webbrowser.open(url)
        return
