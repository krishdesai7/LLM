"""Making sense of malformed tool calls.

Native tool calling keeps calls out of the reply text, but models still drift
over a long run: arguments arrive wrapped in a markdown fence or with a
trailing comma, or a whole call gets written into the reply instead of being
made, as JSON or as the model's own tool-call tokens leaking through. Each is
recoverable, and recovering beats ending the turn on it.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Sequence

__all__ = [
    "Recovered",
    "canonical_name",
    "looks_like_call",
    "parse_arguments",
    "recover_calls",
]

_FENCE = re.compile(r"```[\w-]*[ \t]*\n?(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")
_NAME_PREFIXES = ("functions.", "function.", "tools.", "functions:")

#: Tool-call tokens as Kimi (K2 onwards), DeepSeek and Qwen models emit them;
#: the provider normally turns these into ``tool_calls``, but under pressure
#: they sometimes come through as text.
_KIMI_CALL = re.compile(
    r"<\|tool_call_begin\|>\s*(?P<name>[\w.:-]+?)(?::\d+)?\s*"
    r"<\|tool_call_argument_begin\|>(?P<args>.*?)<\|tool_call_end\|>",
    re.DOTALL,
)
_KIMI_SECTION = re.compile(
    r"<\|tool_calls_section_begin\|>.*?(?:<\|tool_calls_section_end\|>|\Z)", re.DOTALL
)
# DeepSeek's tokens use a full-width bar and a lower-one-eighth block.
_DS = {"bar": "\uff5c", "sp": "\u2581"}
_DEEPSEEK_CALL = re.compile(
    "<{bar}tool{sp}call{sp}begin{bar}>(?:function<{bar}tool{sp}sep{bar}>)?"
    "(?P<name>[\\w.-]+)\\s*(?:<{bar}tool{sp}sep{bar}>)?(?P<args>.*?)"
    "<{bar}tool{sp}call{sp}end{bar}>".format_map(_DS),
    re.DOTALL,
)
_DEEPSEEK_SECTION = re.compile(
    "<{bar}tool{sp}calls{sp}begin{bar}>.*?(?:<{bar}tool{sp}calls{sp}end{bar}>|\\Z)".format_map(
        _DS
    ),
    re.DOTALL,
)
_TAGGED_CALL = re.compile(r"<tool_call>(?P<body>.*?)</tool_call>", re.DOTALL)
_MARKERS = ("<|tool_call", "tool\u2581call", "<tool_call>")


def parse_arguments(text: str) -> tuple[dict[str, Any], bool]:
    """Tool-call arguments as a dict, and whether they needed repairing.

    Repairs a markdown fence, text around the object, trailing commas,
    Python-style quoting, and an object encoded twice. Raises ``ValueError``
    with the original JSON error when none of that helps.
    """
    stripped = text.strip()
    if not stripped:
        return {}, False
    repaired = False
    try:
        value = json.loads(stripped)
    except ValueError as exc:
        value = _repaired(stripped, exc)
        repaired = True
    if isinstance(value, str) and value.strip().startswith("{"):
        value, _ = parse_arguments(value)
        repaired = True
    if not isinstance(value, dict):
        msg = "arguments must be a JSON object"
        # ruff: ignore[type-check-without-type-error] - one error type for callers
        raise ValueError(msg)
    return value, repaired


def _repaired(text: str, original: ValueError) -> Any:
    body = match.group(1) if (match := _FENCE.search(text)) else text
    start, end = body.find("{"), body.rfind("}")
    if start != -1 and end > start:
        body = body[start : end + 1]
    body = _TRAILING_COMMA.sub(r"\1", body)
    try:
        return json.loads(body)
    except ValueError:
        pass
    try:
        return ast.literal_eval(body)
    except ValueError, SyntaxError, MemoryError, RecursionError:
        raise original from None


def canonical_name(name: str, names: Collection[str]) -> str | None:
    """The tool ``name`` refers to, allowing for case, dashes and prefixes."""
    if name in names:
        return name
    plain = name.strip().lower().replace("-", "_")
    for prefix in _NAME_PREFIXES:
        plain = plain.removeprefix(prefix)
    return next((known for known in names if known.lower() == plain), None)


# -- calls written as text --------------------------------------------------


@dataclass(frozen=True, slots=True)
class Recovered:
    """Tool calls found in a reply's text, and the text left without them."""

    #: ``(tool name, arguments as a JSON object string)`` pairs
    calls: list[tuple[str, str]]
    rest: str


def recover_calls(text: str, names: Collection[str]) -> Recovered | None:
    """The tool calls ``text`` writes out instead of making, if any.

    Tool-call tokens and ``<tool_call>`` tags count wherever they appear. Plain
    JSON (fenced or bare) or ``name({...})`` counts only at the very end of
    the reply, since that is where a drifting model puts it, while an answer
    that merely shows an example keeps going after it. Only calls to known
    tools, with readable arguments, are recovered.
    """
    for pattern, section in (
        (_KIMI_CALL, _KIMI_SECTION),
        (_DEEPSEEK_CALL, _DEEPSEEK_SECTION),
    ):
        found = [
            _call(m.group("name"), m.group("args"), names)
            for m in pattern.finditer(text)
        ]
        if found:
            return _complete(found, section.sub("", pattern.sub("", text)))
    tagged = [
        call
        for m in _TAGGED_CALL.finditer(text)
        for call in _calls_in(_loads(m.group("body")), names)
    ]
    if tagged:
        return _complete(tagged, _TAGGED_CALL.sub("", text))
    return _trailing(text, names)


def looks_like_call(text: str, names: Collection[str]) -> bool:
    """Whether ``text`` seems to attempt a tool call, readable or not."""
    if any(marker in text for marker in _MARKERS):
        return True
    alternatives = "|".join(re.escape(name) for name in names)
    if not alternatives:
        return False
    named = rf'"name"\s*:\s*"(?:functions\.)?(?:{alternatives})"'
    spelled = rf"\b(?:{alternatives})\s*\(\s*\{{"
    pattern = f"{named}|{spelled}"
    return re.search(pattern, text) is not None


def _trailing(text: str, names: Collection[str]) -> Recovered | None:
    stripped = text.rstrip()
    fences = list(_FENCE.finditer(stripped))
    if fences and fences[-1].end() == len(stripped):
        last = fences[-1]
        calls = list(_calls_in(_loads(last.group(1)), names))
        if calls:
            return Recovered(calls, stripped[: last.start()].strip())
    for start in _line_starts(stripped, "{["):
        try:
            value, end = json.JSONDecoder().raw_decode(stripped, start)
        except ValueError:
            continue
        if end == len(stripped):
            calls = list(_calls_in(value, names))
            if calls:
                return Recovered(calls, stripped[:start].strip())
            break
    spelled = re.search(
        r"(?:^|\n)[ \t`]*([\w.-]+)\((\{.*\})\)`*\Z", stripped, re.DOTALL
    )
    if spelled:
        name = canonical_name(spelled.group(1), names)
        call = _call(name, spelled.group(2), names) if name else None
        if call:
            return Recovered([call], stripped[: spelled.start()].strip())
    return None


def _line_starts(text: str, openers: str) -> Iterator[int]:
    for index, char in enumerate(text):
        if char in openers and (index == 0 or text[index - 1] == "\n"):
            yield index


def _complete(found: Sequence[tuple[str, str] | None], rest: str) -> Recovered | None:
    """All of ``found``, or nothing: a half-read batch is not safe to run."""
    if not found or any(call is None for call in found):
        return None
    return Recovered([call for call in found if call is not None], rest.strip())


def _call(
    name: str | None, arguments: Any, names: Collection[str]
) -> tuple[str, str] | None:
    known = canonical_name(name, names) if name else None
    if known is None:
        return None
    if isinstance(arguments, dict):
        return known, json.dumps(arguments)
    try:
        parsed, _ = parse_arguments(str(arguments or ""))
    except ValueError:
        return None
    return known, json.dumps(parsed)


def _calls_in(value: Any, names: Collection[str]) -> Iterator[tuple[str, str]]:
    """Calls in a decoded JSON value, in the shapes models write them."""
    if isinstance(value, list):
        for item in value:
            yield from _calls_in(item, names)
        return
    if not isinstance(value, dict):
        return
    if isinstance(value.get("tool_calls"), list):
        yield from _calls_in(value["tool_calls"], names)
        return
    if isinstance(value.get("function"), dict):
        value = value["function"]
    name = next(
        (
            value[key]
            for key in ("name", "tool", "tool_name", "function")
            if isinstance(value.get(key), str)
        ),
        None,
    )
    arguments = next(
        (
            value[key]
            for key in ("arguments", "parameters", "args", "input")
            if key in value
        ),
        {},
    )
    call = _call(name, arguments, names)
    if call is not None:
        yield call


def _loads(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except ValueError:
        try:
            value, _ = parse_arguments(text)
        except ValueError:
            return None
        return value
