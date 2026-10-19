"""Fakes standing in for an OpenAI-compatible endpoint."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from llm.chat import ChatSession
from llm.registry import ModelRef, Provider, Registry

if TYPE_CHECKING:
    from collections.abc import Sequence

PROVIDER = Provider("fake", "FAKE_API_KEY", "https://fake.test/v1")
REF = ModelRef(PROVIDER, "fake-1")


@dataclass
class FakeFunction:
    name: str | None = None
    arguments: str | None = None


@dataclass
class FakeToolCallFragment:
    index: int = 0
    id: str | None = None
    function: FakeFunction | None = None


@dataclass
class FakeDelta:
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[FakeToolCallFragment] | None = None


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
    """Records every request and replays one canned round per call."""

    def __init__(self, rounds: list[list[FakeChunk]]) -> None:
        self.rounds = rounds
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        index = min(len(self.calls), len(self.rounds) - 1)
        self.calls.append(kwargs)
        return FakeStream(self.rounds[index])


class FakeClient:
    def __init__(self, rounds: list[list[FakeChunk]]) -> None:
        self.completions = FakeCompletions(rounds)
        self.chat = type("_Chat", (), {"completions": self.completions})()


class GatedStream:
    """Replays chunks, pausing before any chunk that has a gate."""

    def __init__(self, chunks: list[Any], round_no: int, client: GatedClient) -> None:
        self._chunks = chunks
        self._round = round_no
        self._client = client

    def __iter__(self) -> Any:
        for index, chunk in enumerate(self._chunks):
            key = (self._round, index)
            if key in self._client.gates:
                self._client.reached[key].set()
                assert self._client.gates[key].wait(5), f"gate {key} never opened"
            yield chunk

    def close(self) -> None:
        pass


class GatedClient(FakeClient):
    """A fake client whose streams can be paused at ``(round, chunk)`` gates."""

    def __init__(
        self, rounds: list[list[Any]], gates: Sequence[tuple[int, int]] = ()
    ) -> None:
        super().__init__(rounds)
        self.rounds = rounds
        self.gates = {key: threading.Event() for key in gates}
        self.reached = {key: threading.Event() for key in gates}
        self.completions.create = self._create  # type: ignore[method-assign]
        self._calls = 0

    def _create(self, **_kwargs: Any) -> GatedStream:
        index = min(self._calls, len(self.rounds) - 1)
        self._calls += 1
        return GatedStream(self.rounds[index], index, self)


def text_chunk(text: str) -> FakeChunk:
    return FakeChunk([FakeChoice(FakeDelta(content=text))])


def tool_call_chunks(
    name: str, arguments: str, call_id: str = "call_1"
) -> list[FakeChunk]:
    """A tool call split across chunks, the way providers really stream it."""
    head = FakeToolCallFragment(index=0, id=call_id, function=FakeFunction(name=name))
    chunks = [FakeChunk([FakeChoice(FakeDelta(tool_calls=[head]))])]
    chunks += [
        FakeChunk(
            [
                FakeChoice(
                    FakeDelta(
                        tool_calls=[
                            FakeToolCallFragment(
                                index=0, function=FakeFunction(arguments=piece)
                            )
                        ]
                    )
                )
            ]
        )
        for piece in _split(arguments)
    ]
    return chunks


def _split(text: str, size: int = 4) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def default_round() -> list[FakeChunk]:
    return [
        FakeChunk([FakeChoice(FakeDelta(reasoning_content="hmm"))]),
        text_chunk("Hello"),
        text_chunk(", world"),
        FakeChunk(usage=FakeUsage(prompt_tokens=7, completion_tokens=3)),
    ]


@pytest.fixture
def chunks() -> list[FakeChunk]:
    return default_round()


@pytest.fixture
def make_session() -> Any:
    """Build a ChatSession backed by a fake client.

    Pass ``chunks`` for a single round, or ``rounds`` to script a tool loop.
    """

    def _make(
        chunks: list[FakeChunk] | None = None,
        rounds: list[list[FakeChunk]] | None = None,
        **kwargs: Any,
    ) -> ChatSession:
        if rounds is None:
            rounds = [default_round() if chunks is None else chunks]
        return ChatSession(ref=REF, client=FakeClient(rounds), **kwargs)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def fake_registry() -> Registry:
    return Registry(providers={"fake": PROVIDER}, aliases={})


def calls_of(session: ChatSession) -> list[dict[str, Any]]:
    """The requests a session's fake client received."""
    return session.client.completions.calls  # type: ignore[attr-defined]
