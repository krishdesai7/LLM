"""The local server only answers the page that was launched with its token."""

from __future__ import annotations

import pytest
from litestar import Litestar, get, post
from litestar.middleware import DefineMiddleware
from litestar.testing import TestClient

from llm.gui.security import COOKIE, Guard, make_token

PORT = 8765
LAUNCH_KEY = "launch-key-for-tests"
GOOD = {"cookie": f"{COOKIE}={LAUNCH_KEY}"}


@pytest.fixture
def acted() -> list[int]:
    return []


@pytest.fixture
def client(acted: list[int]) -> TestClient[Litestar]:
    @get("/", sync_to_thread=False)
    def index() -> str:
        return "home"

    @get("/ping", sync_to_thread=False)
    def ping() -> str:
        return "pong"

    @post("/act", sync_to_thread=False)
    def act() -> str:
        acted.append(1)
        return "done"

    app = Litestar(
        [index, ping, act],
        middleware=[DefineMiddleware(Guard, token=LAUNCH_KEY, port=PORT)],
    )
    return TestClient(app, base_url=f"http://127.0.0.1:{PORT}")


def test_make_token_is_long_and_unique() -> None:
    assert len(make_token()) >= 43
    assert make_token() != make_token()


def test_no_cookie_is_403_with_relaunch_hint(client: TestClient[Litestar]) -> None:
    response = client.get("/ping")
    assert response.status_code == 403
    assert "llm gui" in response.text


def test_wrong_cookie_is_403(client: TestClient[Litestar]) -> None:
    assert client.get("/ping", headers={"cookie": f"{COOKIE}=nope"}).status_code == 403


def test_non_ascii_cookie_is_403_not_a_crash(client: TestClient[Litestar]) -> None:
    response = client.get("/ping", headers={b"cookie": f"{COOKIE}=tök€n".encode()})
    assert response.status_code == 403


def test_token_query_sets_cookie_and_redirects(client: TestClient[Litestar]) -> None:
    response = client.get(f"/?token={LAUNCH_KEY}", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert f"{COOKIE}={LAUNCH_KEY}" in response.headers["set-cookie"]


def test_cookie_flags(client: TestClient[Litestar]) -> None:
    cookie = (
        client.get(f"/?token={LAUNCH_KEY}", follow_redirects=False)
        .headers["set-cookie"]
        .lower()
    )
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "path=/" in cookie


def test_wrong_token_query_is_403(client: TestClient[Litestar]) -> None:
    response = client.get("/?token=nope", follow_redirects=False)
    assert response.status_code == 403
    assert "set-cookie" not in response.headers


def test_token_query_only_works_on_the_root_page(client: TestClient[Litestar]) -> None:
    assert client.get(f"/ping?token={LAUNCH_KEY}").status_code == 403


def test_cookie_allows_request(client: TestClient[Litestar]) -> None:
    response = client.get("/ping", headers=GOOD)
    assert (response.status_code, response.text) == (200, "pong")


def test_localhost_host_is_allowed_too(client: TestClient[Litestar]) -> None:
    response = client.get("/ping", headers={**GOOD, "host": f"localhost:{PORT}"})
    assert response.status_code == 200


def test_same_origin_post_is_allowed(
    client: TestClient[Litestar], acted: list[int]
) -> None:
    headers = {**GOOD, "origin": f"http://127.0.0.1:{PORT}"}
    assert client.post("/act", headers=headers).status_code == 201
    assert acted == [1]


def test_foreign_origin_is_403_even_with_a_valid_cookie(
    client: TestClient[Litestar],
) -> None:
    headers = {**GOOD, "origin": "https://evil.example"}
    assert client.get("/ping", headers=headers).status_code == 403


def test_non_loopback_host_is_403_against_dns_rebinding(
    client: TestClient[Litestar],
) -> None:
    headers = {**GOOD, "host": "evil.example"}
    assert client.get("/ping", headers=headers).status_code == 403


def test_wrong_port_in_host_is_403(client: TestClient[Litestar]) -> None:
    headers = {**GOOD, "host": "127.0.0.1:9999"}
    assert client.get("/ping", headers=headers).status_code == 403


def test_a_foreign_origin_post_never_reaches_the_handler(
    client: TestClient[Litestar], acted: list[int]
) -> None:
    headers = {**GOOD, "origin": "https://evil.example"}
    assert client.post("/act", headers=headers).status_code == 403
    assert acted == []
