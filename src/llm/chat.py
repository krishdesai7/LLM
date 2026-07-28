"""Conversation state and streaming, on top of the OpenAI SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Generator

    from openai import OpenAI, Stream
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessageParam,
    )

    from .registry import ModelRef

__all__ = ["ChatSession", "Delta", "Message", "Usage"]

Role = Literal["system", "user", "assistant"]


@dataclass(slots=True)
class Message:
    role: Role
    content: str

    def as_param(self) -> ChatCompletionMessageParam:
        match self.role:
            case "system":
                return {"role": "system", "content": self.content}
            case "assistant":
                return {"role": "assistant", "content": self.content}
            case _:
                return {"role": "user", "content": self.content}


@dataclass(frozen=True, slots=True)
class Delta:
    """One streamed fragment. ``kind`` separates thinking from the answer."""

    kind: Literal["text", "reasoning"]
    text: str


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

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

    def payload(self) -> list[ChatCompletionMessageParam]:
        head = [Message("system", self.system)] if self.system else []
        return [message.as_param() for message in (*head, *self.messages)]

    def send(self, prompt: str) -> str:
        """Non-streaming turn. Returns the assistant's reply."""
        self.messages.append(Message("user", prompt))
        # cast: **params widens the SDK overload to the union return type.
        response = cast(
            "ChatCompletion",
            self.client.chat.completions.create(
                model=self.model_id,
                messages=self.payload(),
                stream=False,
                **self.params,
            ),
        )
        self.last_usage = _usage_of(response)
        reply = (response.choices[0].message.content or "").strip()
        self.messages.append(Message("assistant", reply))
        return reply

    def stream(self, prompt: str) -> Generator[Delta]:
        """Streaming turn. History is updated once the stream is exhausted."""
        self.messages.append(Message("user", prompt))
        chunks: list[str] = []
        stream = cast(
            "Stream[ChatCompletionChunk]",
            self.client.chat.completions.create(
                model=self.model_id,
                messages=self.payload(),
                stream=True,
                **self.params,
            ),
        )
        try:
            for chunk in stream:
                self.last_usage = _usage_of(chunk) or self.last_usage
                for delta in _deltas_of(chunk):
                    if delta.kind == "text":
                        chunks.append(delta.text)
                    yield delta
        finally:
            stream.close()
            self.messages.append(Message("assistant", "".join(chunks).strip()))

    def transcript(self) -> str:
        """The conversation as markdown, for ``/save``."""
        lines = [f"# {self.ref.spec}", ""]
        if self.system:
            lines += ["## system", "", self.system, ""]
        for message in self.messages:
            lines += [f"## {message.role}", "", message.content, ""]
        return "\n".join(lines)


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
