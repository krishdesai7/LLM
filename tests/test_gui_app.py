"""The HTTP API, end to end, over fake model streams."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from conftest import REF, GatedClient, default_round
from litestar.testing import TestClient

from llm import catalog
from llm.chat import ChatSession
from llm.gui.app import create_app
from llm.gui.runs import ChatRun, RunManager
from llm.gui.security import COOKIE
from llm.gui.store import ChatStore, Settings
from llm.registry import Registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from litestar import Litestar

PORT = 8765
LAUNCH_KEY = "launch-key-for-tests"
OWNER = "page-1"


@dataclass
class Script:
    """What the fake model does, and a handle on every client it hands out."""

    rounds: list[list[Any]] = field(default_factory=lambda: [default_round()])
    gates: list[tuple[int, int]] = field(default_factory=list)
    clients: list[GatedClient] = field(default_factory=list)

    def reached(self, key: tuple[int, int]) -> None:
        deadline = time.monotonic() + 5
        while not any(client.reached[key].is_set() for client in self.clients):
            assert time.monotonic() < deadline, f"gate {key} never reached"
            time.sleep(0.005)

    def open(self, key: tuple[int, int]) -> None:
        for client in self.clients:  # chat creation also opens a throwaway client
            client.gates[key].set()


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "sk-test")


@pytest.fixture(autouse=True)
def script(monkeypatch: pytest.MonkeyPatch) -> Script:
    scripted = Script()

    def client_for(_registry: Registry, ref: Any, **_kwargs: Any) -> GatedClient:
        ref.provider.require_api_key()  # what the real client factory does
        client = GatedClient(scripted.rounds, scripted.gates)
        scripted.clients.append(client)
        return client

    monkeypatch.setattr(Registry, "client_for", client_for)
    return scripted


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    return ChatStore(tmp_path / "chats")


@pytest.fixture
def app(fake_registry: Registry, store: ChatStore) -> Litestar:
    return create_app(fake_registry, store, LAUNCH_KEY, PORT)


def _client(app: Litestar, *, cookie: bool = True) -> TestClient[Litestar]:
    client = TestClient(app, base_url=f"http://127.0.0.1:{PORT}")
    if cookie:
        client.headers["cookie"] = f"{COOKIE}={LAUNCH_KEY}"
    return client


@pytest.fixture
def client(app: Litestar) -> TestClient[Litestar]:
    return _client(app)


def _new_chat(client: TestClient[Litestar], **body: Any) -> str:
    response = client.post("/api/chats", json={"model": "fake/fake-1", **body})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _events(response: Any) -> list[dict[str, Any]]:
    blocks = [b for b in response.text.split("\r\n\r\n") if b.strip()]
    return [json.loads(b.removeprefix("data: ")) for b in blocks]


class Background:
    """Run a request on its own thread (a reply blocks until it is finished)."""

    def __init__(self, app: Litestar, call: Callable[[TestClient[Litestar]], Any]):
        self.result: Any = None
        self._thread = threading.Thread(target=self._run, args=(app, call))
        self._thread.start()

    def _run(self, app: Litestar, call: Callable[[TestClient[Litestar]], Any]) -> None:
        self.result = call(_client(app))

    def join(self) -> Any:
        self._thread.join(10)
        assert not self._thread.is_alive()
        return self.result


def _reply(app: Litestar, chat_id: str, prompt: str = "hi") -> Background:
    return Background(
        app,
        lambda c: c.post(
            f"/api/chats/{chat_id}/reply", json={"prompt": prompt, "owner": OWNER}
        ),
    )


# -- chats --------------------------------------------------------------------


def test_create_list_get_delete_round_trip(client: TestClient[Litestar]) -> None:
    chat_id = _new_chat(client, system="be brief", temperature=0.5, web=True)
    listed = client.get("/api/chats").json()
    assert [(c["id"], c["running"], c["model"]) for c in listed] == [
        (chat_id, False, "fake/fake-1")
    ]
    chat = client.get(f"/api/chats/{chat_id}").json()
    assert chat["settings"] == {
        "model": "fake/fake-1",
        "system": "be brief",
        "temperature": 0.5,
        "web": True,
    }
    assert chat["messages"] == []
    assert chat["running"] is False
    assert client.delete(f"/api/chats/{chat_id}").status_code == 204
    assert client.get("/api/chats").json() == []
    assert client.get(f"/api/chats/{chat_id}").status_code == 404


def test_create_uses_the_registry_default_when_no_model_is_given(
    fake_registry: Registry, store: ChatStore
) -> None:
    registry = Registry(
        fake_registry.providers, fake_registry.aliases, default_model="fake/fake-2"
    )
    client = _client(create_app(registry, store, LAUNCH_KEY, PORT))
    response = client.post("/api/chats", json={})
    assert response.status_code == 201
    chat_id = response.json()["id"]
    assert client.get(f"/api/chats/{chat_id}").json()["settings"]["model"] == (
        "fake/fake-2"
    )


def test_create_with_no_model_and_no_default_is_400(
    client: TestClient[Litestar],
) -> None:
    assert client.post("/api/chats", json={}).status_code == 400


def test_create_with_unknown_model_is_400(client: TestClient[Litestar]) -> None:
    response = client.post("/api/chats", json={"model": "nonsense"})
    assert response.status_code == 400
    assert client.get("/api/chats").json() == []


def test_create_with_missing_key_is_400(
    client: TestClient[Litestar], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAKE_API_KEY")
    assert client.post("/api/chats", json={"model": "fake/fake-1"}).status_code == 400


@pytest.mark.parametrize(
    "bad", ["..%2Fx", "a%2Fb", "x.jsonl", "..", "%2e%2e", "nope-not-here"]
)
def test_bad_chat_ids_are_404(client: TestClient[Litestar], bad: str) -> None:
    assert client.get(f"/api/chats/{bad}").status_code == 404
    assert client.delete(f"/api/chats/{bad}").status_code == 404
    assert (
        client.post(
            f"/api/chats/{bad}/reply", json={"prompt": "hi", "owner": OWNER}
        ).status_code
        == 404
    )


def test_settings_can_be_changed_and_persist(client: TestClient[Litestar]) -> None:
    chat_id = _new_chat(client)
    new = {"model": "fake/fake-2", "system": "terse", "temperature": 0.2, "web": True}
    assert client.put(f"/api/chats/{chat_id}/settings", json=new).status_code == 204
    assert client.get(f"/api/chats/{chat_id}").json()["settings"] == new


def test_a_bad_settings_change_is_400_and_changes_nothing(
    client: TestClient[Litestar],
) -> None:
    chat_id = _new_chat(client, system="keep")
    response = client.put(f"/api/chats/{chat_id}/settings", json={"model": "nonsense"})
    assert response.status_code == 400
    settings = client.get(f"/api/chats/{chat_id}").json()["settings"]
    assert (settings["model"], settings["system"]) == ("fake/fake-1", "keep")


# -- replies ------------------------------------------------------------------


def test_reply_streams_events_and_persists(client: TestClient[Litestar]) -> None:
    chat_id = _new_chat(client)
    response = client.post(
        f"/api/chats/{chat_id}/reply", json={"prompt": "hi", "owner": OWNER}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response)
    assert [e["type"] for e in events] == [
        "reasoning",
        "text",
        "text",
        "usage",
        "done",
    ]
    chat = client.get(f"/api/chats/{chat_id}").json()
    assert [(m["role"], m["content"]) for m in chat["messages"]] == [
        ("user", "hi"),
        ("assistant", "Hello, world"),
    ]
    assert chat["messages"][1]["reasoning"] == "hmm"
    assert chat["running"] is False
    assert client.get("/api/chats").json()[0]["title"] == "hi"


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_reply_rejects_an_empty_prompt(
    client: TestClient[Litestar], prompt: str
) -> None:
    chat_id = _new_chat(client)
    response = client.post(
        f"/api/chats/{chat_id}/reply", json={"prompt": prompt, "owner": OWNER}
    )
    assert response.status_code == 400
    assert client.get(f"/api/chats/{chat_id}").json()["messages"] == []


def test_reply_to_a_chat_whose_model_lost_its_key_is_400(
    client: TestClient[Litestar], monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = _new_chat(client)
    monkeypatch.delenv("FAKE_API_KEY")
    response = client.post(
        f"/api/chats/{chat_id}/reply", json={"prompt": "hi", "owner": OWNER}
    )
    assert response.status_code == 400


def test_reply_is_409_while_one_is_running(
    app: Litestar, client: TestClient[Litestar], script: Script
) -> None:
    script.gates = [(0, 0)]
    chat_id = _new_chat(client)
    first = _reply(app, chat_id)
    script.reached((0, 0))
    again = client.post(
        f"/api/chats/{chat_id}/reply", json={"prompt": "again", "owner": OWNER}
    )
    assert again.status_code == 409
    assert client.get("/api/chats").json()[0]["running"] is True
    assert client.get(f"/api/chats/{chat_id}").json()["running"] is True
    script.open((0, 0))
    assert first.join().status_code == 200


def test_settings_are_409_while_a_reply_runs(
    app: Litestar, client: TestClient[Litestar], script: Script
) -> None:
    script.gates = [(0, 0)]
    chat_id = _new_chat(client)
    first = _reply(app, chat_id)
    script.reached((0, 0))
    response = client.put(
        f"/api/chats/{chat_id}/settings", json={"model": "fake/fake-2"}
    )
    assert response.status_code == 409
    script.open((0, 0))
    first.join()


def test_attach_replays_from_after(
    app: Litestar, client: TestClient[Litestar], script: Script
) -> None:
    script.gates = [(0, 1)]
    chat_id = _new_chat(client)
    first = _reply(app, chat_id)
    script.reached((0, 1))
    attached = Background(
        app,
        lambda c: c.post(f"/api/chats/{chat_id}/attach", json={"after": 1}),
    )
    time.sleep(0.1)
    script.open((0, 1))
    full = _events(first.join())
    assert _events(attached.join()) == full[1:]


def test_attach_without_a_running_reply_is_404(client: TestClient[Litestar]) -> None:
    chat_id = _new_chat(client)
    response = client.post(f"/api/chats/{chat_id}/attach", json={"after": 0})
    assert response.status_code == 404


def test_stop_cancels_and_keeps_the_partial_reply(
    app: Litestar, client: TestClient[Litestar], script: Script
) -> None:
    script.gates = [(0, 1)]
    chat_id = _new_chat(client)
    first = _reply(app, chat_id)
    script.reached((0, 1))
    assert client.post(f"/api/chats/{chat_id}/stop").status_code == 204
    script.open((0, 1))
    assert _events(first.join())[-1]["cancelled"] is True
    assert client.get(f"/api/chats/{chat_id}").json()["messages"][-1]["role"] == (
        "assistant"
    )


def test_stop_with_nothing_running_is_a_no_op(client: TestClient[Litestar]) -> None:
    chat_id = _new_chat(client)
    assert client.post(f"/api/chats/{chat_id}/stop").status_code == 204


def test_deleting_a_chat_cancels_its_reply(
    app: Litestar, client: TestClient[Litestar], script: Script
) -> None:
    script.gates = [(0, 1)]
    chat_id = _new_chat(client)
    first = _reply(app, chat_id)
    script.reached((0, 1))
    assert client.delete(f"/api/chats/{chat_id}").status_code == 204
    script.open((0, 1))
    first.join()
    assert client.get("/api/chats").json() == []


def test_a_page_that_stops_heartbeating_has_its_reply_cancelled(
    fake_registry: Registry, store: ChatStore
) -> None:
    """With no stream open and no heartbeat, the background reaper steps in."""
    gated = GatedClient([default_round()], gates=[(0, 0)])
    runs = RunManager(grace_seconds=0.05)
    app = create_app(
        fake_registry, store, LAUNCH_KEY, PORT, runs=runs, reap_interval=0.01
    )
    chat_id = store.create(Settings("fake/fake-1"))
    with _client(app):  # runs the app's lifespan, which owns the reaper
        session = ChatSession(ref=REF, client=gated)  # type: ignore[arg-type]
        run = runs.start(chat_id, session, store, "hi", "page-1")
        deadline = time.monotonic() + 5
        while not run.cancelled:
            assert time.monotonic() < deadline, "reply was never cancelled"
            time.sleep(0.01)
        gated.gates[0, 0].set()
        assert _collect_events(run)[-1]["cancelled"] is True


def _collect_events(run: ChatRun) -> list[dict[str, Any]]:
    async def gather() -> list[dict[str, Any]]:
        return [event async for event in run.subscribe()]

    return asyncio.run(asyncio.wait_for(gather(), timeout=10))


def test_heartbeat_is_204(client: TestClient[Litestar]) -> None:
    assert client.post("/api/heartbeat", json={"owner": OWNER}).status_code == 204


# -- state, models, pages -------------------------------------------------------


def test_state_lists_providers_and_the_search_backend(
    client: TestClient[Litestar],
) -> None:
    state = client.get("/api/state").json()
    assert {"name": "fake", "title": "fake", "ready": True} in state["providers"]
    assert isinstance(state["backend"], str)
    assert state["default_model"] is None


def test_models_come_from_the_catalog(
    client: TestClient[Litestar], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(catalog, "list_models", lambda _provider: ["fake-1", "fake-2"])
    assert client.get("/api/models/fake").json() == ["fake-1", "fake-2"]
    assert client.get("/api/models/nope").status_code == 404


def test_root_serves_the_page_and_static_files_are_served(
    client: TestClient[Litestar],
) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert client.get("/static/index.html").status_code == 200


@pytest.mark.parametrize(
    "path", ["/static/../store.py", "/static/%2e%2e/store.py", "/static/..%2fstore.py"]
)
def test_static_paths_cannot_escape(client: TestClient[Litestar], path: str) -> None:
    response = client.get(path)
    assert response.status_code != 200
    assert "ChatStore" not in response.text


def test_unauthenticated_requests_are_403(app: Litestar) -> None:
    bare = _client(app, cookie=False)
    assert bare.get("/api/chats").status_code == 403
    assert bare.post("/api/chats", json={"model": "fake/fake-1"}).status_code == 403
    assert bare.get("/static/index.html").status_code == 403


def test_a_chat_resumes_with_its_history(
    client: TestClient[Litestar], fake_registry: Registry, store: ChatStore
) -> None:
    chat_id = _new_chat(client)
    client.post(f"/api/chats/{chat_id}/reply", json={"prompt": "hi", "owner": OWNER})
    fresh = _client(create_app(fake_registry, store, LAUNCH_KEY, PORT))
    response = fresh.post(
        f"/api/chats/{chat_id}/reply", json={"prompt": "again", "owner": OWNER}
    )
    assert _events(response)[-1]["type"] == "done"
    roles = [m["role"] for m in fresh.get(f"/api/chats/{chat_id}").json()["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_the_store_directory_is_created_on_first_use(
    fake_registry: Registry, tmp_path: Path
) -> None:
    store = ChatStore(tmp_path / "does" / "not" / "exist")
    client = _client(create_app(fake_registry, store, LAUNCH_KEY, PORT))
    assert client.get("/api/chats").json() == []
    assert _new_chat(client)


def test_the_page_forbids_inline_scripts_and_remote_loads(
    client: TestClient[Litestar],
) -> None:
    page = client.get("/").text
    assert "Content-Security-Policy" in page
    assert "default-src 'none'" in page
    assert "script-src 'self'" in page
    assert "connect-src 'self'" in page
    # every <script> loads a file; none carries code inline
    assert not re.search(r"<script(?![^>]*\bsrc=)", page)


def test_vendored_libraries_match_the_hashes_in_their_readme() -> None:
    vendor = Path(__file__).parents[1] / "src" / "llm" / "gui" / "static" / "vendor"
    rows = re.findall(
        r"\| `([^`]+)` \|.*\| `([^`]+=)` \|", (vendor / "README.md").read_text()
    )
    assert {name for name, _ in rows} == {"marked.umd.js", "purify.min.js"}
    for name, expected in rows:
        digest = base64.b64encode(hashlib.sha256((vendor / name).read_bytes()).digest())
        assert digest.decode() == expected, name


def test_attaching_hands_the_reply_to_the_new_page(
    fake_registry: Registry, store: ChatStore, script: Script
) -> None:
    """After a reload the page has a new id; the reply must follow it."""
    script.gates = [(0, 1)]
    runs = RunManager()
    app = create_app(fake_registry, store, LAUNCH_KEY, PORT, runs=runs)
    client = _client(app)
    chat_id = _new_chat(client)
    first = _reply(app, chat_id)
    script.reached((0, 1))
    assert runs.get(chat_id).owner == OWNER  # type: ignore[union-attr]
    attached = Background(
        app,
        lambda c: c.post(
            f"/api/chats/{chat_id}/attach", json={"after": 0, "owner": "page-2"}
        ),
    )
    deadline = time.monotonic() + 5
    while runs.get(chat_id).owner != "page-2":  # type: ignore[union-attr]
        assert time.monotonic() < deadline, "ownership never moved"
        time.sleep(0.01)
    script.open((0, 1))
    first.join()
    attached.join()
