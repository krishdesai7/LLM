"""Keep other websites, and other hosts, from driving the local server.

Browsers let any page reach ``localhost``, and a rebinding attack can make a
hostile hostname resolve to it. The server therefore answers only requests
that carry the per-launch token (as a cookie), name a loopback ``Host`` on the
right port, and, when they send an ``Origin``, come from that same origin.
This defends against other websites, not other processes on the machine.
"""

from __future__ import annotations

import secrets
from http.cookies import SimpleCookie
from typing import TYPE_CHECKING, cast
from urllib.parse import parse_qs

if TYPE_CHECKING:
    from litestar.types import (
        ASGIApp,
        HTTPResponseBodyEvent,
        HTTPResponseStartEvent,
        HTTPScope,
        Receive,
        Scope,
        Send,
        WebSocketCloseEvent,
    )

__all__ = ["COOKIE", "Guard", "make_token"]

COOKIE = "llm_gui"

_DENIED = (
    b"<!doctype html><title>Forbidden</title>"
    b"<p>This page needs the link printed by <code>llm gui</code>. "
    b"Relaunch it and open that link.</p>"
)


def make_token() -> str:
    """A fresh token with 256 bits of randomness, safe for a URL."""
    return secrets.token_urlsafe(32)


class Guard:
    """ASGI middleware enforcing the rules above on every request."""

    def __init__(self, app: ASGIApp, *, token: str, port: int) -> None:
        self.app = app
        self.token = token
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":
            close = cast(
                "WebSocketCloseEvent", {"type": "websocket.close", "code": 1008}
            )
            await send(close)
            return

        http = cast("HTTPScope", scope)
        headers = {
            name.decode("latin-1").lower(): value.decode("latin-1")
            for name, value in http["headers"]
        }
        host = headers.get("host", "")
        origin = headers.get("origin")
        if host not in self.hosts or (
            origin is not None and origin != f"http://{host}"
        ):
            await self._deny(send)
            return

        query = parse_qs(http["query_string"].decode("latin-1"))
        if http["method"] == "GET" and http["path"] == "/" and "token" in query:
            if self._valid(query["token"][0]):
                await self._grant(send)
            else:
                await self._deny(send)
            return

        if not self._valid(_cookie(headers.get("cookie", ""), COOKIE)):
            await self._deny(send)
            return
        await self.app(scope, receive, send)

    def _valid(self, supplied: str) -> bool:
        return secrets.compare_digest(supplied.encode(), self.token.encode())

    async def _grant(self, send: Send) -> None:
        cookie = f"{COOKIE}={self.token}; HttpOnly; SameSite=Strict; Path=/"
        await send(
            _start(
                303,
                [
                    (b"location", b"/"),
                    (b"set-cookie", cookie.encode()),
                    (b"content-length", b"0"),
                ],
            )
        )
        await send(_body(b""))

    async def _deny(self, send: Send) -> None:
        await send(
            _start(
                403,
                [
                    (b"content-type", b"text/html; charset=utf-8"),
                    (b"content-length", str(len(_DENIED)).encode()),
                ],
            )
        )
        await send(_body(_DENIED))


def _start(status: int, headers: list[tuple[bytes, bytes]]) -> HTTPResponseStartEvent:
    event = {"type": "http.response.start", "status": status, "headers": headers}
    return cast("HTTPResponseStartEvent", event)


def _body(data: bytes) -> HTTPResponseBodyEvent:
    return cast("HTTPResponseBodyEvent", {"type": "http.response.body", "body": data})


def _cookie(header: str, name: str) -> str:
    """One cookie's value from a ``Cookie`` header, or ``""``."""
    jar: SimpleCookie = SimpleCookie()
    try:
        jar.load(header)
    # ruff: ignore[blind-except] - a malformed header is just "no cookie"
    except Exception:
        return ""
    morsel = jar.get(name)
    return morsel.value if morsel is not None else ""
