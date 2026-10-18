"""Conversation state and streaming, on top of the OpenAI SDK.

The tool loop lives here: when a model asks for a tool, the request comes back
as a normal streamed chunk, we run the tool locally, append the output as a
``role: "tool"`` message, and call the API again. The model never touches the
network itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Generator

    from openai import OpenAI, Stream
    from openai.types.chat import (
        ChatCompletionChunk,
        ChatCompletionMessageParam,
    )

    from .registry import ModelRef
    from .tools import ToolBox

__all__ = ["ChatSession", "Delta", "Message", "ToolCall", "Usage"]

Role = Literal["system", "user", "assistant", "tool"]
DeltaKind = Literal["text", "reasoning", "tool_call", "tool_result", "notice"]

#: Ceiling on tool round-trips per turn, so a confused model cannot loop away
#: your tokens forever.
MAX_TOOL_ROUNDS = 6

#: Sent ahead of every conversation. Without the date, models assume it is
#: still their training cutoff and dismiss genuinely recent results as bogus.
PREAMBLE = (
    "Today is {today}. Your training data ends before that, so things may "
    "have changed since; do not assume it is still the year your training "
    "suggests. Answer directly, and say so when you are unsure or when a "
    "question depends on information newer than you have."
)

#: Appended to the preamble while the web tools are on.
WEB_PREAMBLE = (
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

    def as_param(self) -> ChatCompletionMessageParam:
        if self.role == "tool":
            return cast(
                "ChatCompletionMessageParam",
                {
                    "role": "tool",
                    "tool_call_id": self.tool_call_id or "",
                    "content": self.content,
                },
            )
        if self.tool_calls:
            return cast(
                "ChatCompletionMessageParam",
                {
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
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
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
        self.ref = ref
        self.client = client

    def clear(self) -> None:
        self.messages.clear()
        self.last_usage = None

    def system_prompt(self) -> str:
        """The preamble, plus the user's own system prompt if one is set.

        Rebuilt on every request so it tracks ``/web`` and the calendar.
        """
        parts = [PREAMBLE.format(today=_long_date(_today()))]
        if self.tools:
            parts.append(WEB_PREAMBLE)
        if self.system:
            parts.append(self.system)
        return "\n\n".join(parts)

    def payload(self) -> list[ChatCompletionMessageParam]:
        head = Message("system", self.system_prompt())
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
        self.messages.append(Message("user", prompt))
        self.last_usage = None
        for _ in range(MAX_TOOL_ROUNDS):
            calls: list[ToolCall] = []
            yield from self._round(calls)
            if not calls:
                return
            yield from self._run_tools(calls)
        yield Delta("notice", f"stopped after {MAX_TOOL_ROUNDS} rounds of tool calls")

    # -- one API round ------------------------------------------------------

    def _round(self, requested: list[ToolCall]) -> Generator[Delta]:
        """Stream one completion, appending any tool calls to ``requested``."""
        text: list[str] = []
        pending: dict[int, _PartialCall] = {}
        stream = self._create()
        try:
            for chunk in stream:
                self._accumulate_usage(chunk)
                for delta in _deltas_of(chunk):
                    if delta.kind == "text":
                        text.append(delta.text)
                    yield delta
                _collect_tool_calls(chunk, pending)
        finally:
            stream.close()
            calls = [partial.build() for _, partial in sorted(pending.items())]
            requested.extend(calls)
            self.messages.append(Message("assistant", "".join(text).strip(), calls))

    def _create(self) -> Stream[ChatCompletionChunk]:
        extra: dict[str, Any] = {}
        if self.tools:
            extra["tools"] = self.tools.schemas()
            extra["tool_choice"] = "auto"
        # cast: **params widens the SDK overload to the union return type.
        return cast(
            "Stream[ChatCompletionChunk]",
            self.client.chat.completions.create(
                model=self.model_id,
                messages=self.payload(),
                stream=True,
                **extra,
                **self.params,
            ),
        )

    def _accumulate_usage(self, chunk: Any) -> None:
        usage = _usage_of(chunk)
        if usage is None:
            return
        self.last_usage = usage if self.last_usage is None else self.last_usage + usage

    # -- tools --------------------------------------------------------------

    def _run_tools(self, calls: list[ToolCall]) -> Generator[Delta]:
        for call in calls:
            yield Delta("tool_call", call.summary())
            output = (
                self.tools.call(call.name, call.arguments)
                if self.tools is not None
                else f"Error: {call.name} is not available"
            )
            self.messages.append(Message("tool", output, tool_call_id=call.id))
            yield Delta("tool_result", _preview(output))

    def transcript(self) -> str:
        """The conversation as markdown, for ``/save``."""
        lines = [f"# {self.ref.spec}", ""]
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
        partial = pending.setdefault(getattr(fragment, "index", 0) or 0, _PartialCall())
        if getattr(fragment, "id", None):
            partial.id = fragment.id
        function = getattr(fragment, "function", None)
        if function is None:
            continue
        if getattr(function, "name", None):
            partial.name = function.name
        if getattr(function, "arguments", None):
            partial.arguments += function.arguments


def _deltas_of(chunk: Any) -> Generator[Delta]:
    if not getattr(chunk, "choices", None):
        return
    delta = chunk.choices[0].delta
    if delta is None:
        return
    # Several providers stream chain-of-thought on a side channel.
    thinking = getattr(delta, "reasoning_content", None) or getattr(
        delta, "reasoning", None
    )
    if isinstance(thinking, str) and thinking:
        yield Delta("reasoning", thinking)
    if delta.content:
        yield Delta("text", delta.content)


def _usage_of(response: Any) -> Usage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return Usage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def _today() -> date:
    return date.today()


def _long_date(day: date) -> str:
    """``Sunday, October 18, 2026``: unambiguous, unlike 10/18 vs 18/10."""
    return f"{day:%A, %B} {day.day}, {day:%Y}"


def _compact(arguments: str, limit: int = 80) -> str:
    flat = " ".join(arguments.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}…"


def _preview(output: str, limit: int = 120) -> str:
    flat = " ".join(output.split())
    lines = len(output.splitlines())
    head = flat if len(flat) <= limit else f"{flat[:limit]}…"
    return f"{head}  ({len(output)} chars, {lines} lines)"
