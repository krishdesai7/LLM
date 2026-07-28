"""Fakes standing in for an OpenAI-compatible endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from llm.chat import ChatSession
from llm.registry import ModelRef, Provider, Registry

PROVIDER = Provider("fake", "FAKE_API_KEY", "https://fake.test/v1")
REF = ModelRef(PROVIDER, "fake-1")


@dataclass
class FakeDelta:
    content: str | None = None
    reasoning_content: str | None = None


@dataclass
class FakeChoice:
    delta: FakeDelta


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class FakeChunk:
    choices: list[FakeChoice] = field(default_factory=list)
    usage: FakeUsage | None = None


class FakeStream:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self._chunks = chunks
        self.closed = False

    def __iter__(self) -> Any:
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class FakeCompletions:
    """Records every request and replays the same canned chunks."""

    def __init__(self, chunks: list[FakeChunk]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return FakeStream(self.chunks)
        return _as_completion(self.chunks)


class FakeClient:
    def __init__(self, chunks: list[FakeChunk]) -> None:
        self.completions = FakeCompletions(chunks)
        self.chat = type("_Chat", (), {"completions": self.completions})()


def _as_completion(chunks: list[FakeChunk]) -> Any:
    text = "".join(
        choice.delta.content or "" for chunk in chunks for choice in chunk.choices
    )
    message = type("_Message", (), {"content": text})()
    choice = type("_Choice", (), {"message": message})()
    usage = next((chunk.usage for chunk in chunks if chunk.usage), None)
    return type("_Completion", (), {"choices": [choice], "usage": usage})()


def default_chunks() -> list[FakeChunk]:
    return [
        FakeChunk([FakeChoice(FakeDelta(reasoning_content="hmm"))]),
        FakeChunk([FakeChoice(FakeDelta(content="Hello"))]),
        FakeChunk([FakeChoice(FakeDelta(content=", world"))]),
        FakeChunk(usage=FakeUsage(prompt_tokens=7, completion_tokens=3)),
    ]


@pytest.fixture
def chunks() -> list[FakeChunk]:
    return default_chunks()


@pytest.fixture
def make_session() -> Any:
    """Build a ChatSession backed by a fake client."""

    def _make(chunks: list[FakeChunk] | None = None, **kwargs: Any) -> ChatSession:
        client = FakeClient(default_chunks() if chunks is None else chunks)
        return ChatSession(ref=REF, client=client, **kwargs)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def fake_registry() -> Registry:
    return Registry(providers={"fake": PROVIDER}, aliases={})


def calls_of(session: ChatSession) -> list[dict[str, Any]]:
    """The requests a session's fake client received."""
    return session.client.completions.calls  # type: ignore[attr-defined]
