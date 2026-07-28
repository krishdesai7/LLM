"""The interactive chat loop."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from openai import OpenAIError
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.markdown import Markdown

from . import tools, ui
from .registry import LLMError

if TYPE_CHECKING:
    from collections.abc import Callable

    from .chat import ChatSession, Delta
    from .registry import Registry

__all__ = ["run"]

_HISTORY_FILE = Path.home() / ".cache" / "llm" / "history"
_PROMPT_STYLE = Style.from_dict({"prompt": "ansicyan bold"})


@dataclass(slots=True)
class _Loop:
    registry: Registry
    session: ChatSession
    markdown: bool
    show_reasoning: bool
    running: bool = True


def run(
    registry: Registry,
    session: ChatSession,
    *,
    markdown: bool = False,
    show_reasoning: bool = True,
) -> None:
    """Read prompts until EOF, streaming each reply."""
    loop = _Loop(registry, session, markdown, show_reasoning)
    prompt = _prompt_session()
    _banner(loop)

    while loop.running:
        try:
            line = prompt.prompt([("class:prompt", "> ")]).strip()
        except EOFError, KeyboardInterrupt:
            break
        if not line:
            continue
        if line.startswith("/"):
            _run_command(loop, line)
            continue
        _turn(loop, line)

    ui.console.print("[meta]bye[/meta]")


# -- turns ------------------------------------------------------------------


def _turn(loop: _Loop, prompt: str) -> None:
    session = loop.session
    ui.console.print(f"[model]{session.ref.spec}[/model]", style="meta")
    try:
        if loop.markdown:
            _turn_buffered(loop, prompt)
        else:
            _turn_streamed(loop, prompt)
    except KeyboardInterrupt:
        ui.console.print("\n[warn]interrupted[/warn]")
    except (OpenAIError, LLMError) as exc:
        ui.console.print(f"[err]{type(exc).__name__}:[/err] {exc}")
        return
    _footer(loop)


def _turn_streamed(loop: _Loop, prompt: str) -> None:
    thinking = False
    with closing(loop.session.stream(prompt)) as stream:
        for delta in stream:
            if delta.kind == "reasoning":
                if not loop.show_reasoning:
                    continue
                thinking = True
                _write(delta.text, style="reasoning")
                continue
            if thinking:
                thinking = False
                _write("\n\n")
            if delta.kind == "text":
                _write(delta.text)
            else:
                _report(delta)
    ui.console.print("\n")


def _write(text: str, *, style: str | None = None) -> None:
    """Emit a streamed fragment without rich re-wrapping or re-styling it."""
    ui.console.print(
        text, end="", style=style, soft_wrap=True, markup=False, highlight=False
    )


def _report(delta: Delta) -> None:
    """Show tool activity on its own line, out of band from the answer."""
    match delta.kind:
        case "tool_call":
            ui.console.print(f"\n[tool]tool[/tool] {delta.text}", highlight=False)
        case "tool_result":
            ui.console.print(f"[meta]  ↳ {delta.text}[/meta]\n", highlight=False)
        case _:
            ui.console.print(f"[warn]{delta.text}[/warn]")


def _turn_buffered(loop: _Loop, prompt: str) -> None:
    parts: list[str] = []
    with ui.console.status("thinking…"), closing(loop.session.stream(prompt)) as stream:
        for delta in stream:
            if delta.kind == "text":
                parts.append(delta.text)
            elif delta.kind != "reasoning":
                _report(delta)
    ui.console.print(Markdown("".join(parts).strip()))
    ui.console.print()


def _footer(loop: _Loop) -> None:
    usage = loop.session.last_usage
    if usage is not None and usage.total:
        ui.console.print(f"[meta]{usage}[/meta]")


# -- commands ---------------------------------------------------------------


def _cmd_help(_loop: _Loop, _arg: str) -> None:
    ui.console.print(_HELP)


def _cmd_exit(loop: _Loop, _arg: str) -> None:
    loop.running = False


def _cmd_model(loop: _Loop, arg: str) -> None:
    spec = arg.strip() or ui.pick_model(loop.registry)
    if not spec:
        return
    ref, client = loop.registry.open(spec)
    loop.session.switch(ref, client)
    ui.console.print(f"[meta]switched to[/meta] [model]{ref.spec}[/model]")


def _cmd_system(loop: _Loop, arg: str) -> None:
    if not arg.strip():
        current = loop.session.system or "(none)"
        ui.console.print(f"[meta]system prompt:[/meta] {current}")
        return
    loop.session.system = arg.strip()
    ui.console.print("[meta]system prompt set[/meta]")


def _cmd_clear(loop: _Loop, _arg: str) -> None:
    loop.session.clear()
    ui.console.print("[meta]history cleared[/meta]")


def _cmd_history(loop: _Loop, _arg: str) -> None:
    for message in loop.session.messages:
        ui.console.print(f"[meta]{message.role}:[/meta] {message.content}")


def _cmd_save(loop: _Loop, arg: str) -> None:
    path = Path(arg.strip() or "conversation.md").expanduser()
    path.write_text(loop.session.transcript(), encoding="utf-8")
    ui.console.print(f"[meta]saved to[/meta] {path}")


def _cmd_markdown(loop: _Loop, _arg: str) -> None:
    loop.markdown = not loop.markdown
    state = "on (buffered)" if loop.markdown else "off (streaming)"
    ui.console.print(f"[meta]markdown rendering {state}[/meta]")


def _cmd_temp(loop: _Loop, arg: str) -> None:
    try:
        loop.session.params["temperature"] = float(arg)
    except ValueError:
        ui.console.print("[warn]usage: /temp 0.7[/warn]")
        return
    ui.console.print(f"[meta]temperature = {arg}[/meta]")


def _cmd_providers(loop: _Loop, _arg: str) -> None:
    ui.console.print(ui.providers_table(loop.registry))


def _cmd_web(loop: _Loop, _arg: str) -> None:
    """Toggle the web tools for this conversation."""
    if loop.session.tools:
        loop.session.tools = None
        ui.console.print("[meta]web access off[/meta]")
        return
    loop.session.tools = tools.web_toolbox()
    ui.console.print(f"[meta]web access on — {ui.search_status()}[/meta]")


_COMMANDS: dict[str, Callable[[_Loop, str], None]] = {
    "help": _cmd_help,
    "?": _cmd_help,
    "exit": _cmd_exit,
    "quit": _cmd_exit,
    "model": _cmd_model,
    "m": _cmd_model,
    "system": _cmd_system,
    "clear": _cmd_clear,
    "new": _cmd_clear,
    "history": _cmd_history,
    "save": _cmd_save,
    "markdown": _cmd_markdown,
    "temp": _cmd_temp,
    "providers": _cmd_providers,
    "web": _cmd_web,
}

_HELP = """[bold]commands[/bold]
  [prompt]/model[/prompt] [meta][spec][/meta]   switch model (no arg opens the picker)
  [prompt]/system[/prompt] [meta][text][/meta]  show or set the system prompt
  [prompt]/clear[/prompt]           forget the conversation so far
  [prompt]/history[/prompt]         replay the conversation
  [prompt]/save[/prompt] [meta][path][/meta]    write the transcript as markdown
  [prompt]/web[/prompt]             toggle web search + page fetching
  [prompt]/markdown[/prompt]        toggle rendered markdown vs. live streaming
  [prompt]/temp[/prompt] [meta]0.7[/meta]       set sampling temperature
  [prompt]/providers[/prompt]       which providers have keys
  [prompt]/exit[/prompt]            quit  ([meta]ctrl-d[/meta] also works)

[meta]enter sends · alt+enter (or esc then enter) inserts a newline[/meta]"""


def _run_command(loop: _Loop, line: str) -> None:
    name, _, arg = line[1:].partition(" ")
    handler = _COMMANDS.get(name.lower())
    if handler is None:
        ui.console.print(f"[warn]unknown command /{name}[/warn] — try /help")
        return
    handler(loop, arg)


# -- input ------------------------------------------------------------------


def _prompt_session() -> PromptSession[str]:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event: object) -> None:
        event.current_buffer.validate_and_handle()  # pyrefly: ignore[missing-attribute]

    @bindings.add("escape", "enter")
    def _newline(event: object) -> None:
        event.current_buffer.insert_text("\n")  # pyrefly: ignore[missing-attribute]

    history = None
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(_HISTORY_FILE))
    except OSError:
        pass

    return PromptSession(
        history=history,
        multiline=True,
        key_bindings=bindings,
        style=_PROMPT_STYLE,
    )


def _banner(loop: _Loop) -> None:
    session = loop.session
    ui.console.print(
        f"[model]{session.ref.spec}[/model] "
        f"[meta]via {session.ref.provider.base_url}[/meta]"
    )
    if session.system:
        ui.console.print(f"[meta]system:[/meta] {session.system}")
    if session.tools:
        ui.console.print(f"[meta]web access on — {ui.search_status()}[/meta]")
    ui.console.print("[meta]/help for commands · ctrl-d to quit[/meta]\n")
