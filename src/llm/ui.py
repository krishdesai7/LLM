"""Terminal bits: the model picker and shared rendering helpers."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

import questionary
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from . import catalog, web

if TYPE_CHECKING:
    from .registry import Provider, Registry

__all__ = [
    "console",
    "pick_model",
    "providers_table",
    "search_status",
    "search_table",
    "stderr",
]

_THEME = Theme(
    {
        "prompt": "bold cyan",
        "model": "bold magenta",
        "reasoning": "dim italic",
        "meta": "dim",
        "tool": "bold green",
        "warn": "yellow",
        "err": "bold red",
    }
)

console = Console(theme=_THEME)
#: for progress chatter that must not pollute a piped answer
stderr = Console(theme=_THEME, stderr=True)

_STYLE = questionary.Style(
    [
        ("qmark", "fg:cyan bold"),
        ("question", "bold"),
        ("answer", "fg:magenta bold"),
        ("highlighted", "fg:cyan bold"),
        ("pointer", "fg:cyan bold"),
    ]
)

_BACK = object()
_CUSTOM = object()


def pick_model(registry: Registry, *, refresh: bool = False) -> str | None:
    """Interactive model dropdown. Returns a model spec, or ``None`` if aborted."""
    while True:
        picked = _pick_source(registry)
        if picked is None:
            return None
        if picked is _CUSTOM:
            return _ask_free_text()
        if isinstance(picked, str):
            return picked

        model_id = _pick_from_provider(picked, refresh=refresh)
        if model_id is _BACK:
            continue
        if not isinstance(model_id, str):
            return None
        return model_id if "/" in model_id else f"{picked.name}/{model_id}"


def _pick_source(registry: Registry) -> str | Provider | Any | None:
    ready = registry.ready_providers()
    if not ready:
        console.print(
            "[warn]No provider API keys found in the environment.[/warn] "
            "Set one (e.g. MOONSHOT_API_KEY) and try again.",
        )
        return None

    choices: list[Any] = [
        questionary.Choice(title=f"{name}   ({target})", value=target)
        for name, target in sorted(registry.aliases.items())
    ]
    if choices:
        choices.append(questionary.Separator())

    choices.extend(
        questionary.Choice(title=f"{p.title}  ·  browse models", value=p) for p in ready
    )
    choices.extend(
        (
            questionary.Separator(),
            questionary.Choice(title="type a model id…", value=_CUSTOM),
        )
    )

    missing = len(registry.providers) - len(ready)
    if missing:
        choices.append(
            questionary.Separator(f"  ({missing} more providers need an API key)")
        )
    return _select("Model", choices)


def _pick_from_provider(provider: Provider, *, refresh: bool) -> str | Any | None:
    with console.status(f"Fetching models from {provider.title}…"):
        models = catalog.list_models(provider, refresh=refresh)

    if not models:
        console.print(
            f"[warn]{provider.title} did not return a model list.[/warn] "
            "Enter the id yourself:"
        )
        return _ask_free_text()

    choices: list[Any] = [
        questionary.Choice(title=model, value=model) for model in models
    ]
    choices.extend(
        (
            questionary.Separator(),
            questionary.Choice(title="back", value=_BACK),
        )
    )
    return _select(f"{provider.title} model", choices, searchable=True)


def _ask_free_text() -> str | None:
    answer = questionary.text(
        "Model id (alias, 'provider/model', or a known id):", style=_STYLE
    ).ask()
    return answer.strip() if answer and answer.strip() else None


def _select(message: str, choices: list[Any], *, searchable: bool = False) -> Any:
    kwargs: dict[str, Any] = {"style": _STYLE, "use_shortcuts": False}
    if searchable:
        kwargs["use_search_filter"] = True
        kwargs["use_jk_keys"] = False
    try:
        return questionary.select(message, choices=choices, **kwargs).ask()
    except TypeError:  # older questionary without the search-filter options
        return questionary.select(message, choices=choices, style=_STYLE).ask()


def search_status() -> str:
    """One-line description of which search backend is in play."""
    try:
        backend = web.active_backend()
    except web.WebError as exc:
        return str(exc)
    if not backend.is_ready():
        return f"{backend.label} selected but not configured"
    return f"searching via {backend.label}"


def search_table() -> Table:
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("backend")
    table.add_column("env var")
    table.add_column("ready", justify="center")
    active = None
    with suppress(web.WebError):
        active = web.active_backend().name
    for backend in web.backends():
        marker = " [tool]←active[/tool]" if backend.name == active else ""
        table.add_row(
            f"[model]{backend.name}[/model]{marker}",
            backend.url_var or backend.env_var or "[meta]none needed[/meta]",
            "[green]✓[/green]" if backend.is_ready() else "[meta]—[/meta]",
        )
    return table


def providers_table(registry: Registry) -> Table:
    table = Table(box=None, pad_edge=False, header_style="bold")
    table.add_column("provider")
    table.add_column("env var")
    table.add_column("base url", style="meta")
    table.add_column("key", justify="center")
    for provider in registry.sorted_providers():
        ready = provider.is_ready()
        table.add_row(
            f"[model]{provider.name}[/model]",
            provider.env_var,
            provider.base_url,
            "[green]✓[/green]" if ready else "[meta]—[/meta]",
        )
    return table
