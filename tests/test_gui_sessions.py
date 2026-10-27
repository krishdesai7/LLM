"""Turning stored chats and settings into live ``ChatSession`` objects."""

from __future__ import annotations

import pytest

from llm.chat import Message
from llm.gui.sessions import apply_settings, build_session
from llm.gui.store import Settings, StoredChat
from llm.registry import LLMError, Registry


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "sk-test")


def _stored(settings: Settings, messages: list[Message] | None = None) -> StoredChat:
    return StoredChat("c1", 0.0, 0.0, settings, messages or [])


def test_build_restores_messages_and_settings(fake_registry: Registry) -> None:
    history = [Message("user", "hi"), Message("assistant", "hello")]
    session = build_session(
        fake_registry, _stored(Settings("fake/fake-1", "be brief", 0.3), history)
    )
    assert session.ref.spec == "fake/fake-1"
    assert session.system == "be brief"
    assert session.params == {"temperature": 0.3}
    assert session.tools is None
    assert session.messages == history


def test_build_does_not_alias_the_stored_messages(fake_registry: Registry) -> None:
    stored = _stored(Settings("fake/fake-1"), [Message("user", "hi")])
    session = build_session(fake_registry, stored)
    session.messages.append(Message("assistant", "x"))
    assert len(stored.messages) == 1


def test_temperature_none_leaves_params_empty(fake_registry: Registry) -> None:
    session = build_session(fake_registry, _stored(Settings("fake/fake-1")))
    assert session.params == {}


def test_web_true_attaches_toolbox_and_false_clears_it(
    fake_registry: Registry,
) -> None:
    session = build_session(fake_registry, _stored(Settings("fake/fake-1", web=True)))
    assert session.tools is not None
    assert session.tools.names == ["web_search", "fetch_url", "find_in_page"]
    apply_settings(session, fake_registry, Settings("fake/fake-1", web=False))
    assert session.tools is None


def test_apply_settings_switches_model_keeping_history(
    fake_registry: Registry,
) -> None:
    session = build_session(
        fake_registry, _stored(Settings("fake/fake-1"), [Message("user", "hi")])
    )
    apply_settings(
        session, fake_registry, Settings("fake/fake-2", "new system", 0.9, True)
    )
    assert session.model_id == "fake-2"
    assert session.system == "new system"
    assert session.params == {"temperature": 0.9}
    assert session.tools is not None
    assert [m.content for m in session.messages] == ["hi"]


def test_apply_settings_clears_temperature_when_none(fake_registry: Registry) -> None:
    session = build_session(fake_registry, _stored(Settings("fake/fake-1", None, 0.5)))
    apply_settings(session, fake_registry, Settings("fake/fake-1"))
    assert session.params == {}


def test_unknown_model_raises_llm_error(fake_registry: Registry) -> None:
    with pytest.raises(LLMError):
        build_session(fake_registry, _stored(Settings("nonsense")))


def test_failed_apply_leaves_the_session_untouched(fake_registry: Registry) -> None:
    session = build_session(fake_registry, _stored(Settings("fake/fake-1", "keep")))
    with pytest.raises(LLMError):
        apply_settings(session, fake_registry, Settings("nonsense", "changed"))
    assert session.system == "keep"
    assert session.model_id == "fake-1"


def test_missing_key_raises_llm_error(
    fake_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAKE_API_KEY")
    with pytest.raises(LLMError):
        build_session(fake_registry, _stored(Settings("fake/fake-1")))
