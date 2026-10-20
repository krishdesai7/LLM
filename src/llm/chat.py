"""Conversation state and streaming, on top of the OpenAI SDK.

The tool loop lives here: when a model asks for a tool, the request comes back
as a normal streamed chunk, we run the tool locally, append the output as a
``role: "tool"`` message, and call the API again. The model never touches the
network itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Final, LiteralString

    from openai import OpenAI, Stream
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionMessageParam,
    )

    from .registry import ModelRef
    from .tools import ToolBox

__all__: list[str] = ["ChatSession", "Delta", "Message", "ToolCall", "Usage"]

Role: type[Role] = Literal["system", "user", "assistant", "tool"]
DeltaKind: type[DeltaKind] = Literal[
    "text", "reasoning", "tool_call", "tool_result", "notice"
]

#: Ceiling on tool round-trips per turn, so a confused model cannot loop away
#: your tokens forever.
MAX_TOOL_ROUNDS: Final[int] = 6

#: Sent ahead of every conversation. Without the date, models assume it is
#: still their training cutoff and dismiss genuinely recent results as bogus.
PREAMBLE: Final[LiteralString] = (
    "Today is {today}. Your training data ends before that, so things may "
    "have changed since; do not assume it is still the year your training "
    "suggests. Answer directly, and say so when you are unsure or when a "
    "question depends on information newer than you have."
)

#: Appended to the preamble while the web tools are on.
WEB_PREAMBLE: Final[LiteralString] = (
    "You have web_search and fetch_url tools that see the live web as of "
    "today. Dates in their results that are later than your training data "
    "are real, not errors: trust them over your own sense of the present. Use "
    "the tools for anything recent or that you would otherwise be guessing "
    "at, and skip them for stable facts. One or two targeted searches are "
    "usually enough; prefer reading an authoritative source (official docs, "
    "release notes, the project's own site) over searching again. Cite the "
    "URLs you relied on, and say when they are old enough that the answer "
    "may have changed since."
)


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_param(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    def summary(self) -> str:
        return f"{self.name}({_compact(self.arguments)})"


@dataclass(slots=True)
class Message:
    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    #: Chain-of-thought for display and storage only; never sent back.
    reasoning: str = ""
    reasoning_seconds: float = 0.0

    def as_param(self) -> ChatCompletionMessageParam:
        if self.role == "tool":
            return cast(
                typ="ChatCompletionMessageParam",
                val={
                    "role": "tool",
                    "tool_call_id": self.tool_call_id or "",
                    "content": self.content,
                },
            )
        if self.tool_calls:
            return cast(
                typ="ChatCompletionMessageParam",
                val={
                    "role": "assistant",
                    "content": self.content or None,
                    "tool_calls": [call.as_param() for call in self.tool_calls],
                },
            )
        match self.role:
            case "system":
                return {"role": "system", "content": self.content}
            case "assistant":
                return {"role": "assistant", "content": self.content}
            case _:
                return {"role": "user", "content": self.content}


@dataclass(frozen=True, slots=True)
class Delta:
    """One streamed fragment. ``kind`` says what the caller should do with it."""

    kind: DeltaKind
    text: str


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def __str__(self) -> str:
        return (
            f"{self.prompt_tokens} in / {self.completion_tokens} out "
            f"= {self.total} tokens"
        )


@dataclass(slots=True)
class ChatSession:
    """A single conversation, pinned to one model at a time."""

    ref: ModelRef
    client: OpenAI
    system: str | None = None
    messages: list[Message] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    tools: ToolBox | None = None
    last_usage: Usage | None = None

    @property
    def model_id(self) -> str:
        return self.ref.model_id

    def switch(self, ref: ModelRef, client: OpenAI) -> None:
        """Point the conversation at a different model, keeping history."""
        self.ref: ModelRef = ref
        self.client: OpenAI = client

    def clear(self) -> None:
        self.messages.clear()
        self.last_usage = None

    def system_prompt(self) -> str:
        """The preamble, plus the user's own system prompt if one is set.

        Rebuilt on every request so it tracks ``/web`` and the calendar.
        """
        parts: list[str] = [PREAMBLE.format(today=_long_date(day=_today()))]
        if self.tools:
            parts.append(WEB_PREAMBLE)
        if self.system:
            parts.append(self.system)
        return "\n\n".join(parts)

    def payload(self) -> list[ChatCompletionMessageParam]:
        head = Message(role="system", content=self.system_prompt())
        return [message.as_param() for message in (head, *self.messages)]

    def send(self, prompt: str) -> str:
        """Run a whole turn and return just the reply text."""
        return "".join(
            delta.text for delta in self.stream(prompt) if delta.kind == "text"
        )

    def stream(self, prompt: str) -> Generator[Delta]:
        """Stream a turn, running any tools the model asks for along the way.

        History is updated as each round completes, so an abandoned stream
        still leaves the conversation coherent.
        """
        self.messages.append(Message(role="user", content=prompt))
        self.last_usage = None
        for _ in range(MAX_TOOL_ROUNDS):
            calls: list[ToolCall] = []
            yield from self._round(requested=calls)
            if not calls:
                return
            yield from self._run_tools(calls)
        yield Delta(
            kind="notice", text=f"stopped after {MAX_TOOL_ROUNDS} rounds of tool calls"
        )

    # -- one API round ------------------------------------------------------

    def _round(self, requested: list[ToolCall]) -> Generator[Delta]:
        """Stream one completion, appending any tool calls to ``requested``."""
        text: list[str] = []
        thinking: list[str] = []
        watch = _Stopwatch()
        pending: dict[int, _PartialCall] = {}
        stream: Stream[ChatCompletionChunk] = self._create()
        try:
            for chunk in stream:
                self._accumulate_usage(chunk)
                for delta in _deltas_of(chunk):
                    if delta.kind == "reasoning":
                        thinking.append(delta.text)
                        watch.start()
                    elif delta.kind == "text":
                        text.append(delta.text)
                        watch.stop()
                    yield delta
                _collect_tool_calls(chunk, pending)
        finally:
            stream.close()
            calls: list[ToolCall] = [
                partial.build() for _, partial in sorted(pending.items())
            ]
            requested.extend(calls)
            watch.stop()
            self.messages.append(
                Message(
                    role="assistant",
                    content="".join(text).strip(),
                    tool_calls=calls,
                    reasoning="".join(thinking),
                    reasoning_seconds=watch.elapsed,
                )
            )

    def _create(self) -> Stream[ChatCompletionChunk]:
        extra: dict[str, Any] = {}
        if self.tools:
            extra["tools"] = self.tools.schemas()
            extra["tool_choice"] = "auto"
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=self.payload(),
            stream=True,
            **extra,
            **self.params,
        )

    def _accumulate_usage(self, chunk: Any) -> None:
        usage: Usage | None = _usage_of(response=chunk)
        if usage is None:
            return
        self.last_usage: Usage = (
            usage if self.last_usage is None else self.last_usage + usage
        )

    # -- tools --------------------------------------------------------------

    def _run_tools(self, calls: list[ToolCall]) -> Generator[Delta]:
        for call in calls:
            yield Delta(kind="tool_call", text=call.summary())
            output: str = (
                self.tools.call(call.name, call.arguments)
                if self.tools is not None
                else f"Error: {call.name} is not available"
            )
            self.messages.append(
                Message(role="tool", content=output, tool_call_id=call.id)
            )
            yield Delta(kind="tool_result", text=_preview(output))

    def transcript(self) -> str:
        """The conversation as markdown, for ``/save``."""
        lines: list[str] = [f"# {self.ref.spec}", ""]
        if self.system:
            lines += ["## system", "", self.system, ""]
        for message in self.messages:
            lines += [f"## {message.role}", ""]
            for call in message.tool_calls:
                lines += [f"- called `{call.summary()}`"]
            if message.content:
                lines += [message.content]
            lines += [""]
        return "\n".join(lines)


@dataclass(slots=True)
class _Stopwatch:
    """Times the reasoning phase of one round: first thought to first answer."""

    started: float | None = None
    elapsed: float = 0.0
    stopped: bool = False

    def start(self) -> None:
        if self.started is None:
            self.started: float = _now()

    def stop(self) -> None:
        if self.started is not None and not self.stopped:
            self.elapsed: float = _now() - self.started
            self.stopped = True


@dataclass(slots=True)
class _PartialCall:
    """A tool call being assembled from streamed fragments."""

    id: str = ""
    name: str = ""
    arguments: str = ""

    def build(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments=self.arguments)


def _collect_tool_calls(chunk: Any, pending: dict[int, _PartialCall]) -> None:
    """Merge this chunk's tool-call fragments into ``pending``.

    Providers stream a call in pieces: the id and name arrive once, then the
    arguments accumulate character by character under the same index.
    """
    if not getattr(chunk, "choices", None):
        return
    delta = chunk.choices[0].delta
    for fragment in getattr(delta, "tool_calls", None) or []:
        partial: _PartialCall = pending.setdefault(
            getattr(fragment, "index", 0) or 0, _PartialCall()
        )
        if getattr(fragment, "id", None):
            partial.id = fragment.id
        function: Any | None = getattr(fragment, "function", None)
        if function is None:
            continue
        if getattr(function, "name", None):
            partial.name = function.name
        if getattr(function, "arguments", None):
            partial.arguments += function.arguments


def _deltas_of(chunk: Any) -> Generator[Delta]:
    if not getattr(chunk, "choices", None):
        return
    delta: Any = chunk.choices[0].delta
    if delta is None:
        return
    # Several providers stream chain-of-thought on a side channel.
    thinking: Any | None = getattr(delta, "reasoning_content", None) or getattr(
        delta, "reasoning", None
    )
    if isinstance(thinking, str) and thinking:
        yield Delta(kind="reasoning", text=thinking)
    if delta.content:
        yield Delta(kind="text", text=delta.content)


def _usage_of(response: Any) -> Usage | None:
    usage: Any | None = getattr(response, "usage", None)
    if usage is None:
        return None
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _today() -> date:
    return date.today()


def _now() -> float:
    return time.monotonic()


def _long_date(day: date) -> str:
    """``Sunday, October 18, 2026``: unambiguous, unlike 10/18 vs 18/10."""
    return f"{day:%A, %B} {day.day}, {day:%Y}"


def _compact(arguments: str, limit: int = 80) -> str:
    flat: str = " ".join(arguments.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}…"


def _preview(output: str, limit: int = 120) -> str:
    flat: str = " ".join(output.split())
    lines: int = len(output.splitlines())
    head: str = flat if len(flat) <= limit else f"{flat[:limit]}…"
    return f"{head}  ({len(output)} chars, {lines} lines)"
