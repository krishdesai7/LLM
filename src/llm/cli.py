"""``llm`` command line: pick a model, everything else is inferred."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer
from openai import OpenAIError

from . import catalog, reddit, repl, tools, ui
from . import web as webmod
from .chat import ChatSession
from .registry import LLMError, Registry, config_path

if TYPE_CHECKING:
    from openai import OpenAI

    from .registry import ModelRef

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
    help="Chat with any OpenAI-compatible model. Name the model; "
    "the API key and base URL are inferred.",
)

ModelOpt = Annotated[
    str | None,
    typer.Option(
        "--model",
        "-m",
        help="Alias, 'provider/model', or a known model id. Omit to pick one.",
    ),
]
SystemOpt = Annotated[str | None, typer.Option("--system", "-s", help="System prompt.")]
TempOpt = Annotated[
    float | None, typer.Option("--temperature", "-t", help="Sampling temperature.")
]
MarkdownOpt = Annotated[
    bool,
    typer.Option(
        "--markdown/--stream",
        help="Render the reply as markdown once complete, instead of streaming.",
    ),
]
WebOpt = Annotated[
    bool,
    typer.Option(
        "--web/--no-web",
        help="Let the model search the web and fetch pages via tool calls.",
    ),
]


@app.callback()
def main_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


@app.command()
def chat(
    model: ModelOpt = None,
    system: SystemOpt = None,
    temperature: TempOpt = None,
    markdown: MarkdownOpt = False,
    web: WebOpt = False,
    no_reasoning: Annotated[
        bool, typer.Option("--no-reasoning", help="Hide streamed reasoning traces.")
    ] = False,
) -> None:
    """Start an interactive chat session."""
    registry = Registry.load()
    ref, client = _open(registry, model)
    session = ChatSession(
        ref=ref,
        client=client,
        system=system,
        params=_params(temperature),
        tools=tools.web_toolbox() if web else None,
    )
    repl.run(registry, session, markdown=markdown, show_reasoning=not no_reasoning)


@app.command()
def ask(
    prompt: Annotated[list[str] | None, typer.Argument(help="The prompt.")] = None,
    model: ModelOpt = None,
    system: SystemOpt = None,
    temperature: TempOpt = None,
    markdown: MarkdownOpt = False,
    web: WebOpt = False,
) -> None:
    """One-shot prompt. Reads stdin when piped, so `cat x | llm ask` works."""
    text = " ".join(prompt or []).strip()
    piped = "" if sys.stdin.isatty() else sys.stdin.read().strip()
    full = "\n\n".join(part for part in (text, piped) if part)
    if not full:
        raise typer.BadParameter("no prompt given")

    registry = Registry.load()
    ref, client = _open(registry, model, interactive=False)
    session = ChatSession(
        ref=ref,
        client=client,
        system=system,
        params=_params(temperature),
        tools=tools.web_toolbox() if web else None,
    )
    try:
        if markdown:
            from rich.markdown import Markdown

            ui.console.print(Markdown(session.send(full)))
        else:
            _echo_stream(session, full)
    except (OpenAIError, LLMError) as exc:
        _die(exc)


def _echo_stream(session: ChatSession, prompt: str) -> None:
    """Stream an answer to stdout, reporting tool use on stderr."""
    for delta in session.stream(prompt):
        if delta.kind == "text":
            ui.console.print(
                delta.text, end="", soft_wrap=True, markup=False, highlight=False
            )
        elif delta.kind == "tool_call":
            ui.stderr.print(f"[tool]tool[/tool] {delta.text}", highlight=False)
    ui.console.print()


@app.command()
def gui(
    model: ModelOpt = None,
    port: Annotated[int, typer.Option("--port", help="Port on 127.0.0.1.")] = 8765,
    no_open: Annotated[
        bool, typer.Option("--no-open", help="Print the link; don't open a browser.")
    ] = False,
) -> None:
    """Chat in the browser: past chats, collapsible thinking, rendered markdown."""
    from .gui import launch

    try:
        launch.serve(Registry.load(), model=model, port=port, open_browser=not no_open)
    except LLMError as exc:
        _die(exc)


@app.command()
def models(
    provider: Annotated[
        str | None, typer.Argument(help="Provider name. Omit for every ready one.")
    ] = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Bypass the 24h cache.")
    ] = False,
) -> None:
    """List the models each provider currently serves."""
    registry = Registry.load()
    try:
        targets = (
            [registry.provider(provider)] if provider else registry.ready_providers()
        )
    except LLMError as exc:
        _die(exc)

    if not targets:
        ui.console.print("[warn]No provider API keys found.[/warn] Try: llm providers")
        raise typer.Exit(1)

    for target in targets:
        found = catalog.list_models(target, refresh=refresh)
        ui.console.print(f"\n[model]{target.title}[/model] [meta]{target.name}/[/meta]")
        if not found:
            ui.console.print("  [meta]no model list available[/meta]")
        for model_id in found:
            ui.console.print(f"  {model_id}")


@app.command()
def providers() -> None:
    """Show every provider, its env var, base URL, and whether the key is set."""
    registry = Registry.load()
    ui.console.print(ui.providers_table(registry))
    if registry.aliases:
        ui.console.print("\n[bold]aliases[/bold]")
        for name, target in sorted(registry.aliases.items()):
            ui.console.print(f"  [model]{name}[/model] → {target}")
    ui.console.print(f"\n[meta]config: {config_path()}[/meta]")


@app.command()
def resolve(
    spec: Annotated[str, typer.Argument(help="Model spec to resolve.")],
) -> None:
    """Explain how a model spec maps to a provider, key, and base URL."""
    registry = Registry.load()
    try:
        ref = registry.resolve(spec)
    except LLMError as exc:
        _die(exc)
    ready = "set" if ref.provider.is_ready() else "[err]missing[/err]"
    ui.console.print(f"[model]{spec}[/model] →")
    ui.console.print(f"  provider  {ref.provider.name} ({ref.provider.title})")
    ui.console.print(f"  model     {ref.model_id}")
    ui.console.print(f"  base_url  {ref.provider.base_url}")
    ui.console.print(f"  api_key   {ref.provider.env_var} ({ready})")


@app.command("web")
def web_command(
    query: Annotated[
        list[str] | None, typer.Argument(help="Search query. Omit to list backends.")
    ] = None,
) -> None:
    """Show the search backends, or run a query through the active one."""
    if not query:
        ui.console.print(ui.search_table())
        ui.console.print(f"\n[meta]{ui.search_status()}[/meta]")
        ui.console.print(f"[meta]{reddit.status()}[/meta]")
        ui.console.print(
            "[meta]override with LLM_SEARCH_BACKEND=<name>[/meta]",
        )
        return
    try:
        results = webmod.search(" ".join(query))
    except webmod.WebError as exc:
        _die(exc)
    ui.console.print(webmod.render(results), highlight=False)


@app.command()
def fetch(
    url: Annotated[str, typer.Argument(help="A public http(s) URL.")],
    chars: Annotated[
        int | None,
        typer.Option(
            "--chars", help="Truncate the extracted text.", show_default=False
        ),
    ] = None,
) -> None:
    """Print a page as the model would see it. Useful for checking extraction."""
    try:
        ui.console.print(webmod.fetch(url, max_chars=chars), highlight=False)
    except webmod.WebError as exc:
        _die(exc)


@app.command()
def init() -> None:
    """Write a starter config file with alias and provider examples."""
    path = config_path()
    if path.exists():
        ui.console.print(f"[warn]already exists:[/warn] {path}")
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_CONFIG_TEMPLATE, encoding="utf-8")
    ui.console.print(f"[meta]wrote[/meta] {path}")


@app.command("clear-cache")
def clear_cache() -> None:
    """Drop the cached model lists."""
    ui.console.print(f"[meta]removed {catalog.clear_cache()} cached list(s)[/meta]")


# -- helpers ----------------------------------------------------------------


def _open(
    registry: Registry, model: str | None, *, interactive: bool = True
) -> tuple[ModelRef, OpenAI]:
    spec = model or registry.default_model
    if not spec and interactive:
        spec = ui.pick_model(registry)
    if not spec:
        ui.console.print(
            "[err]no model selected.[/err] Pass --model, or set default_model in "
            f"{config_path()}"
        )
        raise typer.Exit(1)
    try:
        return registry.open(spec)
    except LLMError as exc:
        _die(exc)


def _params(temperature: float | None) -> dict[str, float]:
    return {} if temperature is None else {"temperature": temperature}


def _die(exc: Exception) -> NoReturn:
    ui.console.print(f"[err]{exc}[/err]")
    raise typer.Exit(1) from exc


_CONFIG_TEMPLATE = """\
# llm config — https://platform.openai.com/docs/libraries
#
# Aliases are the names you see in the picker; the right-hand side is
# "provider/model-id". A bare model id whose prefix is recognised (gpt-,
# claude-, kimi-, deepseek-, grok-, glm-, gemini-, ...) needs no alias.
#
# Run `llm models <provider>` to see the ids a provider actually serves today,
# then paste them below.

# default_model = "moonshot/kimi-k2"

[aliases]
# kimi = "moonshot/kimi-k2"
# r1 = "deepseek/deepseek-reasoner"

# Any OpenAI-compatible endpoint can be added or overridden here.
# [providers.myserver]
# env_var = "MYSERVER_API_KEY"
# base_url = "https://llm.internal.example.com/v1"
# label = "Internal"
# headers = { "X-Title" = "llm-cli" }
"""
