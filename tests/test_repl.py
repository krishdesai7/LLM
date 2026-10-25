from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm import repl


class _ScriptedPrompt:
    """Stands in for the prompt_toolkit session: replays canned input."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = iter(lines)

    def prompt(self, _message: object) -> str:
        try:
            return next(self._lines)
        except StopIteration:
            raise EOFError from None


@pytest.fixture
def drive(monkeypatch, capsys, fake_registry) -> Any:
    """Run the REPL over scripted input and return everything it printed."""

    def _drive(lines: list[str], session: Any) -> str:
        monkeypatch.setattr(repl, "_prompt_session", lambda: _ScriptedPrompt(lines))
        repl.run(fake_registry, session)
        return capsys.readouterr().out

    return _drive


def test_a_turn_streams_the_reply(drive: Any, make_session: Any) -> None:
    assert "Hello, world" in drive(["hi"], make_session())


def test_reasoning_is_shown_by_default(drive: Any, make_session: Any) -> None:
    assert "hmm" in drive(["hi"], make_session())


def test_blank_lines_are_ignored(drive: Any, make_session: Any) -> None:
    session = make_session()
    drive(["", "   ", "hi"], session)
    assert len([m for m in session.messages if m.role == "user"]) == 1


def test_system_command_sets_the_prompt(drive: Any, make_session: Any) -> None:
    session = make_session()
    drive(["/system be terse", "hi"], session)
    assert session.system == "be terse"
    assert session.payload()[0]["role"] == "system"


def test_clear_command_drops_history(drive: Any, make_session: Any) -> None:
    session = make_session()
    drive(["hi", "/clear"], session)
    assert session.messages == []


def test_temp_command_sets_a_param(drive: Any, make_session: Any) -> None:
    session = make_session()
    drive(["/temp 0.3"], session)
    assert session.params["temperature"] == pytest.approx(0.3)


def test_bad_temp_is_reported_not_raised(drive: Any, make_session: Any) -> None:
    session = make_session()
    out = drive(["/temp hot"], session)
    assert "usage: /temp" in out
    assert "temperature" not in session.params


def test_unknown_command_is_reported(drive: Any, make_session: Any) -> None:
    assert "unknown command" in drive(["/nope"], make_session())


def test_exit_command_stops_the_loop(drive: Any, make_session: Any) -> None:
    session = make_session()
    drive(["/exit", "hi"], session)
    assert session.messages == []


def test_save_writes_a_transcript(drive: Any, make_session: Any, tmp_path) -> None:
    target = tmp_path / "out.md"
    session = make_session()
    drive(["hi", f"/save {target}"], session)
    assert "Hello, world" in target.read_text(encoding="utf-8")


def test_api_errors_do_not_kill_the_session(
    drive: Any, make_session: Any, monkeypatch
) -> None:
    from openai import APIError

    session = make_session()

    def boom(**_kwargs: Any) -> Any:
        raise APIError("upstream exploded", request=None, body=None)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(session.client.completions, "create", boom)
    out = drive(["hi", "/clear"], session)
    assert "upstream exploded" in out
    assert "history cleared" in out


def test_history_file_honours_xdg_state_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    assert repl.history_file() == tmp_path / "llm" / "history"


def test_history_file_defaults_to_local_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    assert repl.history_file() == Path.home() / ".local" / "state" / "llm" / "history"
