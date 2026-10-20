"""Live model discovery.

Model ids drift constantly, so nothing here is hardcoded: the list comes from
each provider's ``GET /models`` endpoint and is cached on disk so the picker
stays instant.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openai._client import OpenAI
from openai.pagination import SyncPage
from openai.types.model import Model

if TYPE_CHECKING:
    from typing import Literal

    from .registry import Provider

__all__: list[str] = ["cache_dir", "clear_cache", "list_models"]

CACHE_TTL_SECONDS: Literal[86400] = 24 * 60 * 60


def cache_dir() -> Path:
    base: str | None = os.environ.get(key="XDG_CACHE_HOME")
    root: Path = Path(base).expanduser() if base else Path.home() / ".cache"
    return root / "llm" / "models"


def list_models(
    provider: Provider,
    *,
    refresh: bool = False,
    ttl: float = CACHE_TTL_SECONDS,
) -> list[str]:
    """Model ids offered by ``provider``, newest cache first.

    Returns an empty list if the provider has no key, has no ``/models``
    endpoint, or is unreachable — callers fall back to free-text entry.
    """
    if not provider.is_ready():
        return []

    cached: list[str] | None = None if refresh else _read_cache(provider.name, ttl)
    if cached is not None:
        return cached

    models: list[str] = _fetch(provider)
    if models:
        _write_cache(provider.name, models)
    return models


def clear_cache() -> int:
    """Delete every cached model list. Returns the number of files removed."""
    directory: Path = cache_dir()
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.glob(pattern="*.json"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def _fetch(provider: Provider) -> list[str]:
    from .registry import make_client

    try:
        client: OpenAI = make_client(provider, timeout=15.0, max_retries=0)
        page: SyncPage[Model] = client.models.list()
        ids: set[str] = {
            _normalise(model_id=model.id)
            for model in page
            if getattr(model, "id", None)
        }
    except Exception:  # ruff: ignore[blind-except] - discovery is best-effort by design
        return []
    return sorted(ids)


def _normalise(model_id: str) -> str:
    # Gemini's OpenAI-compatible endpoint returns "models/gemini-..." ids.
    return model_id.removeprefix("models/")


def _cache_file(provider_name: str) -> Path:
    return cache_dir() / f"{provider_name}.json"


def _read_cache(provider_name: str, ttl: float) -> list[str] | None:
    path: Path = _cache_file(provider_name)
    try:
        age: float = time.time() - path.stat().st_mtime
        if age > ttl:
            return None
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return data
    return None


def _write_cache(provider_name: str, models: list[str]) -> None:
    path: Path = _cache_file(provider_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data=json.dumps(obj=models), encoding="utf-8")
    except OSError:
        pass  # a cold cache is never fatal
