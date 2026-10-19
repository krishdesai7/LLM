"""``llm gui``: launching the server and wiring the app from the environment."""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING, Any

import pytest
from litestar.testing import TestClient
from typer.testing import CliRunner

from llm.cli import app
from llm.gui import launch
from llm.gui.security import COOKIE
from llm.registry import Registry, UnknownModelError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace the real server and browser with recorders."""
    seen: dict[str, list[Any]] = {"server": [], "browser": []}

    def record_open(_port: int, url: str) -> None:
        seen["browser"].append(url)

    monkeypatch.setattr(launch, "_run_server", seen["server"].append)
    monkeypatch.setattr(launch, "_open_when_ready", record_open)
    return seen


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """No user config, and chats go to a temp directory."""
    monkeypatch.setenv("LLM_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    yield tmp_path
    for key in launch.ENV_KEYS:  # serve() leaves its settings in the environment
        os.environ.pop(key, None)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_gui_command_is_registered_and_shows_help() -> None:
    result = CliRunner().invoke(app, ["gui", "--help"])
    assert result.exit_code == 0
    for flag in ("--model", "--port", "--no-open"):
        assert flag in result.output


def test_other_commands_are_still_there() -> None:
    result = CliRunner().invoke(app, ["--help"])
    for name in ("chat", "ask", "models", "providers", "web", "gui"):
        assert name in result.output


def test_serve_prints_a_tokenised_url_and_starts_the_server(
    calls: dict[str, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    port = _free_port()
    launch.serve(Registry.load(), model=None, port=port, open_browser=False)
    assert calls["server"] == [port]
    printed = capsys.readouterr().out
    assert f"http://127.0.0.1:{port}/?token=" in printed


def test_serve_opens_the_same_url_it_prints(
    calls: dict[str, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    launch.serve(Registry.load(), model=None, port=_free_port(), open_browser=True)
    assert len(calls["browser"]) == 1
    assert calls["browser"][0] in capsys.readouterr().out


def test_serve_does_not_open_a_browser_when_told_not_to(
    calls: dict[str, list[Any]],
) -> None:
    launch.serve(Registry.load(), model=None, port=_free_port(), open_browser=False)
    assert calls["browser"] == []


def test_serve_reports_a_busy_port_before_starting(calls: dict[str, list[Any]]) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        with pytest.raises(launch.PortBusyError, match=str(port)):
            launch.serve(Registry.load(), model=None, port=port, open_browser=True)
    assert calls == {"server": [], "browser": []}


def test_serve_rejects_an_unknown_model_before_starting(
    calls: dict[str, list[Any]],
) -> None:
    with pytest.raises(UnknownModelError):
        launch.serve(
            Registry.load(), model="nonsense", port=_free_port(), open_browser=False
        )
    assert calls["server"] == []


def test_the_cli_reports_launch_errors_without_a_traceback() -> None:
    result = CliRunner().invoke(app, ["gui", "--model", "nonsense", "--no-open"])
    assert result.exit_code == 1
    assert "nonsense" in result.output
    assert "Traceback" not in result.output


def test_the_app_factory_builds_a_guarded_app_from_the_environment() -> None:
    port = _free_port()
    launch.serve(
        Registry.load(), model="moonshot/kimi-k2", port=port, open_browser=False
    )
    served = launch.app_factory()
    client = TestClient(served, base_url=f"http://127.0.0.1:{port}")
    assert client.get("/api/state").status_code == 403
    token = os.environ[launch.ENV_TOKEN]
    client.headers["cookie"] = f"{COOKIE}={token}"
    state = client.get("/api/state").json()
    assert state["default_model"] == "moonshot/kimi-k2"
