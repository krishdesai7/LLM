"""Conversation state and streaming, on top of the OpenAI SDK.

The tool loop lives here: when a model asks for a tool, the request comes back
as a normal streamed chunk, we run the tool locally, append the output as a
``role: "tool"`` message, and call the API again. The model never touches the
network itself.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Literal, cast

from . import repair

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
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

#: Tool calls one turn may make. Research fans out (a thread, then each place
#: it names, then a fallback source for each), so this is generous; it is a
#: ceiling so a confused model cannot loop away your tokens forever.
TOOL_BUDGET: Final[int] = 40

#: From this many calls left, tool results remind the model to wrap up.
BUDGET_WARNING: Final[int] = 5

#: Every this many calls, a tool result restates the rules that matter most.
#: Guidance at the top of a long context loses its pull as the context grows;
#: a few lines near the end, where the model is looking, keep it on track.
REMIND_EVERY: Final[int] = 8

REMINDER: Final[LiteralString] = (
    "[Reminder: make tool calls through the tool-calling interface, never as "
    "text in your reply. Keep a line of notes on what you have found and what "
    "is still open. When a call fails, follow its advice instead of retrying "
    "variations; when every route to something fails, note the dead end and "
    "move on.]"
)

#: Sent, once a turn, when a reply tried to write a tool call as text and it
#: could not be read: one nudge rather than ending the turn on garbage.
CORRECTION: Final[LiteralString] = (
    "Your last reply wrote a tool call out as text instead of making it, so it "
    "did not run, and it could not be read as a call. If you meant to use a "
    "tool, call it through the tool-calling interface, with its arguments as "
    "a JSON object. Otherwise, give your answer."
)

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
    "at, and skip them for stable facts.\n\n"
    "Match the effort to the question. A simple lookup needs a search or two; "
    "prefer reading an authoritative source (official docs, release notes, "
    "the project's own site) over searching again. A question with many parts "
    "(say, every option a thread recommends, each checked against its own "
    "site and its reviews) needs many calls: you can make up to {budget} per "
    "turn, several at once when they don't depend on each other. For that "
    "kind of question:\n"
    "- First work out what needs covering, then go through it.\n"
    "- Make tool calls only through the tool-calling interface. A call "
    "written out as text in your reply does not run.\n"
    "- Keep notes as you go: alongside your tool calls, write a line or two "
    "on what the last results told you and what is still open. Older tool "
    "results are shortened as newer ones arrive, so your notes are how "
    "findings survive. Calling a tool again with the same arguments brings a "
    "shortened result back in full, without a new request.\n"
    "- Long pages come a part at a time. Read on with fetch_url's offset, "
    "or use find_in_page to go straight to what you need (on a menu, say, "
    "'vegetarian').\n"
    "- When a call fails, read why. The error says whether retrying can help; "
    "if it can't, get the information another way (a different site, a "
    "search, a listing or review page) rather than repeating the call. Not "
    "every lead pans out: note the dead end and move on.\n"
    "- Weigh what you read: a highly upvoted comment counts for more than a "
    "lone one, and a primary source beats a second-hand account of it.\n\n"
    "Cite the URLs you relied on, and say when they are old enough that the "
    "answer may have changed since. Say plainly what you could not verify."
)

#: Tool results this many characters long or shorter are always sent whole.
ELIDE_OVER: Final[int] = 2_000

#: How much tool output, counting back from the newest, is sent whole. Older
#: long results go as a short stub instead (history keeps them in full): a
#: research turn piles up pages fast, and a long context of stale page text
#: costs tokens on every round and dulls the model's attention to what matters
#: now. Generous, since the models this is for have large windows.
RECENT_TOOL_CHARS: Final[int] = 120_000


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
    #: Chain-of-thought. Shown and stored; sent back only to models that need
    #: it (see :data:`llm.registry.ReasoningReplay`).
    reasoning: str = ""
    reasoning_seconds: float = 0.0

    def as_param(self, *, with_reasoning: bool = False) -> ChatCompletionMessageParam:
        """This message as the API expects it.

        ``with_reasoning`` adds an assistant message's ``reasoning_content``,
        even if empty: a model that requires it rejects a message without one.
        """
        param = self._param()
        if with_reasoning and self.role == "assistant":
            cast("dict[str, Any]", param)["reasoning_content"] = self.reasoning
        return param

    def _param(self) -> ChatCompletionMessageParam:
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
    tool_budget: int = TOOL_BUDGET
    last_usage: Usage | None = None
    #: What the harness did to keep the model on track last turn, by kind
    last_events: Counter[str] = field(default_factory=Counter)

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
        self.last_events.clear()
        if self.tools is not None:
            self.tools.reset()

    def system_prompt(self) -> str:
        """The preamble, plus the user's own system prompt if one is set.

        Rebuilt on every request so it tracks ``/web`` and the calendar.
        """
        parts: list[str] = [PREAMBLE.format(today=_long_date(day=_today()))]
        if self.tools:
            parts.append(WEB_PREAMBLE.format(budget=self.tool_budget))
        if self.system:
            parts.append(self.system)
        return "\n\n".join(parts)

    def payload(
        self, extra: Sequence[Message] = ()
    ) -> list[ChatCompletionMessageParam]:
        """What to send: the system prompt, then history as the model should see it.

        ``extra`` goes on the end for this request only, without entering
        history.
        """
        head = Message(role="system", content=self.system_prompt())
        sent = _shortened([*self.messages, *extra], keep=RECENT_TOOL_CHARS)
        replay = self.ref.reasoning_replay
        turn = next(
            (
                i
                for i in reversed(range(len(self.messages)))
                if self.messages[i].role == "user"
            ),
            0,
        )
        return [
            head.as_param(),
            *(
                message.as_param(
                    with_reasoning=replay == "all" or (replay == "turn" and i > turn)
                )
                for i, message in enumerate(sent)
            ),
        ]

    def send(self, prompt: str) -> str:
        """Run a whole turn and return just the reply text."""
        return "".join(
            delta.text for delta in self.stream(prompt) if delta.kind == "text"
        )

    def stream(self, prompt: str) -> Generator[Delta]:
        """Stream a turn, running any tools the model asks for along the way.

        Once ``tool_budget`` calls have run, the model gets one more round with
        tools switched off, so a long search still ends in an answer. A turn
        in which the harness had to step in ends with a notice counting how.

        History is updated as each round completes, so an abandoned stream
        still leaves the conversation coherent.
        """
        self.messages.append(Message(role="user", content=prompt))
        self.last_usage = None
        self.last_events = Counter()
        if self.tools is not None:
            self.tools.start_turn()
        yield from self._turn()
        if self.tools is not None:
            self.last_events.update(self.tools.events)
        if self.last_events:
            yield Delta(kind="notice", text=_summary(self.last_events))

    def _turn(self) -> Generator[Delta]:
        left: int = self.tool_budget
        correction: list[Message] = []
        corrected = False
        while True:
            calls: list[ToolCall] = []
            yield from self._round(requested=calls, final=left <= 0, extra=correction)
            correction = []
            if not calls and left > 0 and self.tools:
                if self._recover(into=calls):
                    self.last_events["recovered"] += len(calls)
                    yield Delta(
                        kind="notice",
                        text=f"recovered {len(calls)} tool "
                        f"call{'s' if len(calls) > 1 else ''} written as text",
                    )
                elif not corrected and repair.looks_like_call(
                    self.messages[-1].content, self.tools.names
                ):
                    corrected = True
                    self.last_events["corrected"] += 1
                    correction = [self.messages.pop(), Message("user", CORRECTION)]
                    yield Delta(
                        kind="notice",
                        text="the reply wrote a tool call as text that couldn't be "
                        "read; asking again",
                    )
                    continue
            if not calls:
                return
            if left <= 0:
                yield from self._run_tools(calls, budget=0)
                yield Delta(
                    kind="notice",
                    text="stopped: the model kept asking for tools after its budget",
                )
                return
            yield from self._run_tools(calls, budget=left)
            left = max(0, left - len(calls))
            if left <= 0:
                yield Delta(
                    kind="notice",
                    text=f"used all {self.tool_budget} tool calls; asking for a "
                    "final answer",
                )

    def _recover(self, into: list[ToolCall]) -> bool:
        """Turn tool calls the last reply wrote as text into real ones.

        The reply is rewritten in history as a proper call, so later requests
        show the model a well-formed call of its own rather than the drift.
        """
        message = self.messages[-1]
        if self.tools is None:
            return False
        found = repair.recover_calls(message.content, self.tools.names)
        if found is None:
            return False
        base = len(self.messages)
        message.content = found.rest
        message.tool_calls = [
            ToolCall(f"recovered_{base}_{i}", name, arguments)
            for i, (name, arguments) in enumerate(found.calls)
        ]
        into.extend(message.tool_calls)
        return True

    # -- one API round ------------------------------------------------------

    def _round(
        self,
        requested: list[ToolCall],
        *,
        final: bool = False,
        extra: Sequence[Message] = (),
    ) -> Generator[Delta]:
        """Stream one completion, appending any tool calls to ``requested``.

        ``final`` keeps the tools listed (history refers to them) but tells
        the model not to call any. ``extra`` is sent after history, this once.
        """
        text: list[str] = []
        thinking: list[str] = []
        watch = _Stopwatch()
        pending: dict[int, _PartialCall] = {}
        stream: Stream[ChatCompletionChunk] = self._create(final=final, extra=extra)
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

    def _create(
        self, *, final: bool = False, extra: Sequence[Message] = ()
    ) -> Stream[ChatCompletionChunk]:
        options: dict[str, Any] = {}
        if self.tools:
            options["tools"] = self.tools.schemas()
            options["tool_choice"] = "none" if final else "auto"
        return self.client.chat.completions.create(
            model=self.model_id,
            messages=self.payload(extra),
            stream=True,
            **options,
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

    def _run_tools(self, calls: list[ToolCall], *, budget: int) -> Generator[Delta]:
        """Run as many of ``calls`` as ``budget`` allows.

        Every call gets an answer, since the API rejects unanswered ones;
        those past the budget are answered without running.
        """
        for index, call in enumerate(calls):
            yield Delta(kind="tool_call", text=call.summary())
            if index >= budget:
                output = "Error: not run, because this turn's tool budget is used up."
            elif self.tools is None:
                output = f"Error: {call.name} is not available"
            else:
                output = self.tools.call(call.name, call.arguments)
            self.messages.append(
                Message(role="tool", content=output, tool_call_id=call.id)
            )
            yield Delta(kind="tool_result", text=_preview(output))
        left = max(0, budget - len(calls))
        made = self.tool_budget - budget
        ran = min(len(calls), budget)
        notes: list[str] = []
        if left > 0 and (made + ran) // REMIND_EVERY > made // REMIND_EVERY:
            notes.append(REMINDER)
        if left <= BUDGET_WARNING:
            notes.append(self._budget_note(left))
        if notes:
            self.messages[-1].content += "\n\n" + "\n\n".join(notes)

    def _budget_note(self, left: int) -> str:
        if left > 0:
            return (
                f"[{left} of {self.tool_budget} tool calls left this turn. Spend "
                "them on what matters most, then answer.]"
            )
        return (
            f"[That was the last of this turn's {self.tool_budget} tool calls; no "
            "more tools can run. Write your answer now from what you have found, "
            "and say plainly what you could not verify or did not get to.]"
        )

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


def _shortened(messages: list[Message], *, keep: int) -> list[Message]:
    """``messages`` with old, long tool results cut to a stub, for sending.

    Counting back from the newest, tool output is kept whole until ``keep``
    characters have gone by; past that, results over ``ELIDE_OVER`` become a
    stub saying what the call was and how to get it back. Results after the
    last assistant message are kept regardless: the model hasn't read them.
    Once a result is shortened it stays shortened, so the prefix providers
    cache stays stable from one round to the next.
    """
    calls = {call.id: call for message in messages for call in message.tool_calls}
    unread = next(
        (
            i + 1
            for i in reversed(range(len(messages)))
            if messages[i].role == "assistant"
        ),
        0,
    )
    sent = list(messages)
    seen = 0
    for index in reversed(range(len(messages))):
        message = messages[index]
        if message.role != "tool":
            continue
        size = len(message.content)
        seen += size
        if index < unread and seen > keep and size > ELIDE_OVER:
            call = calls.get(message.tool_call_id or "")
            sent[index] = replace(message, content=_stub(message.content, call))
    return sent


def _stub(content: str, call: ToolCall | None) -> str:
    what = call.summary() if call is not None else "an earlier tool call"
    head = _compact(content.split("\n", 1)[0], limit=160)
    return (
        f"[Result of {what}: {len(content):,} characters, starting "
        f"{head!r}. Shortened to save room now that newer results have "
        "arrived; make the same call again to see it in full, which needs no "
        "new request.]"
    )


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


#: How each kind of intervention reads in the end-of-turn notice.
_EVENTS: Final[dict[str, tuple[str, str]]] = {
    "recovered": ("tool call recovered from text", "tool calls recovered from text"),
    "corrected": ("garbled call asked for again", "garbled calls asked for again"),
    "repaired": ("malformed call repaired", "malformed calls repaired"),
    "refused": ("repeated call not run", "repeated calls not run"),
    "stalled": ("no-progress warning", "no-progress warnings"),
}


def _summary(events: Counter[str]) -> str:
    """``harness: 2 tool calls recovered from text, 1 no-progress warning``."""
    parts = [
        f"{events[key]} {one if events[key] == 1 else many}"
        for key, (one, many) in _EVENTS.items()
        if events[key]
    ]
    return "harness: " + ", ".join(parts)


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
