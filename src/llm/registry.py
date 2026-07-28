"""Model -> (API key, base URL) resolution.

Everything downstream only ever names a *model*. This module turns that name
into a fully configured OpenAI-compatible client by looking up the provider
that serves it, reading the provider's API key out of the environment, and
pointing the SDK at the provider's base URL.

Resolution order for a model spec:

1. an alias from the user config      (``kimi`` -> ``moonshot/kimi-k2``)
2. an explicit ``provider/model`` pair (``groq/llama-3.3-70b``)
3. a known model-id prefix            (``kimi-k3`` -> moonshot)
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from openai import OpenAI

__all__ = [
    "MissingAPIKeyError",
    "ModelRef",
    "Provider",
    "Registry",
    "UnknownModelError",
    "config_path",
    "make_client",
]


class LLMError(Exception):
    """Base class for errors raised by this package."""


class UnknownModelError(LLMError):
    """The model spec could not be attributed to any known provider."""


class MissingAPIKeyError(LLMError):
    """The provider was resolved but its API key is not in the environment."""


@dataclass(frozen=True, slots=True)
class Provider:
    """An OpenAI-compatible API endpoint."""

    name: str
    env_var: str
    base_url: str
    label: str = ""
    #: used when the endpoint needs no real credential (local servers)
    fallback_key: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.label or self.name

    def api_key(self) -> str | None:
        """The credential for this provider, or ``None`` if unavailable."""
        key = os.environ.get(self.env_var, "").strip()
        return key or self.fallback_key

    def is_ready(self) -> bool:
        return self.api_key() is not None

    def require_api_key(self) -> str:
        key = self.api_key()
        if key is None:
            msg = (
                f"{self.title} needs an API key: set {self.env_var} in your "
                f"environment (or in .env)."
            )
            raise MissingAPIKeyError(msg)
        return key


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A concrete model on a concrete provider."""

    provider: Provider
    model_id: str

    @property
    def spec(self) -> str:
        return f"{self.provider.name}/{self.model_id}"

    def __str__(self) -> str:
        return self.spec


# --------------------------------------------------------------------------
# Built-in providers. Add your own in config.toml under [providers.<name>].
# --------------------------------------------------------------------------

_BUILTIN_PROVIDERS: tuple[Provider, ...] = (
    Provider("openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "OpenAI"),
    Provider(
        "anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com/v1", "Anthropic"
    ),
    Provider(
        "google",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "Google Gemini",
    ),
    Provider(
        "moonshot", "MOONSHOT_API_KEY", "https://api.moonshot.ai/v1", "Moonshot (Kimi)"
    ),
    Provider("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "DeepSeek"),
    Provider("xai", "XAI_API_KEY", "https://api.x.ai/v1", "xAI (Grok)"),
    Provider("zai", "ZAI_API_KEY", "https://api.z.ai/api/paas/v4", "Z.ai (GLM)"),
    Provider("mistral", "MISTRAL_API_KEY", "https://api.mistral.ai/v1", "Mistral"),
    Provider("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "Groq"),
    Provider("cerebras", "CEREBRAS_API_KEY", "https://api.cerebras.ai/v1", "Cerebras"),
    Provider("together", "TOGETHER_API_KEY", "https://api.together.xyz/v1", "Together"),
    Provider(
        "fireworks",
        "FIREWORKS_API_KEY",
        "https://api.fireworks.ai/inference/v1",
        "Fireworks",
    ),
    Provider(
        "deepinfra",
        "DEEPINFRA_API_KEY",
        "https://api.deepinfra.com/v1/openai",
        "DeepInfra",
    ),
    Provider(
        "perplexity", "PERPLEXITY_API_KEY", "https://api.perplexity.ai", "Perplexity"
    ),
    Provider(
        "openrouter",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        "OpenRouter",
    ),
    Provider(
        "ollama",
        "OLLAMA_API_KEY",
        "http://localhost:11434/v1",
        "Ollama (local)",
        fallback_key="ollama",
    ),
    Provider(
        "lmstudio",
        "LMSTUDIO_API_KEY",
        "http://localhost:1234/v1",
        "LM Studio (local)",
        fallback_key="lm-studio",
    ),
)

#: Model-id prefixes that identify a provider without an explicit ``provider/``.
#: Longest match wins, so ordering here is irrelevant.
_PREFIX_HINTS: Mapping[str, str] = {
    "gpt-": "openai",
    "chatgpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "claude-": "anthropic",
    "gemini-": "google",
    "kimi": "moonshot",
    "moonshot": "moonshot",
    "deepseek": "deepseek",
    "grok": "xai",
    "glm-": "zai",
    "mistral": "mistral",
    "ministral": "mistral",
    "magistral": "mistral",
    "codestral": "mistral",
    "devstral": "mistral",
    "pixtral": "mistral",
    "sonar": "perplexity",
}


def config_path() -> Path:
    """Location of the user config file (honours ``$LLM_CONFIG``/XDG)."""
    override = os.environ.get("LLM_CONFIG")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "llm" / "config.toml"


@dataclass(frozen=True, slots=True)
class Registry:
    """Providers + aliases, merged from the built-ins and the user config."""

    providers: Mapping[str, Provider]
    aliases: Mapping[str, str]
    default_model: str | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> Self:
        """Build a registry, layering ``config.toml`` over the built-ins."""
        providers = {p.name: p for p in _BUILTIN_PROVIDERS}
        data = _read_config(path if path is not None else config_path())

        for name, raw in _table(data, "providers").items():
            providers[name] = _provider_from_config(name, raw, providers.get(name))

        aliases = {
            str(k): str(v) for k, v in _table(data, "aliases").items() if str(v).strip()
        }
        default = data.get("default_model")
        return cls(
            providers=providers,
            aliases=aliases,
            default_model=str(default) if default else None,
        )

    # -- lookup ------------------------------------------------------------

    def provider(self, name: str) -> Provider:
        try:
            return self.providers[name]
        except KeyError:
            known = ", ".join(sorted(self.providers))
            msg = f"unknown provider {name!r}; known providers: {known}"
            raise UnknownModelError(msg) from None

    def ready_providers(self) -> list[Provider]:
        """Providers whose API key is present in the environment."""
        return [p for p in self.sorted_providers() if p.is_ready()]

    def sorted_providers(self) -> list[Provider]:
        return sorted(self.providers.values(), key=lambda p: p.name)

    def resolve(self, spec: str) -> ModelRef:
        """Turn a model spec into a :class:`ModelRef`.

        Accepts an alias, ``provider/model-id``, or a bare model id whose
        prefix identifies the provider.
        """
        name = spec.strip()
        if not name:
            msg = "no model given"
            raise UnknownModelError(msg)

        for _ in range(len(self.aliases) + 1):
            target = self.aliases.get(name)
            if target is None or target == name:
                break
            name = target.strip()

        if "/" in name:
            head, _, tail = name.partition("/")
            if head in self.providers and tail:
                return ModelRef(self.providers[head], tail)

        hint = _prefix_hint(name)
        if hint is not None and hint in self.providers:
            return ModelRef(self.providers[hint], name)

        msg = (
            f"cannot tell which provider serves {spec!r}. "
            f"Use 'provider/model' (e.g. 'moonshot/{name}'), or add an alias "
            f"to {config_path()}."
        )
        raise UnknownModelError(msg)

    # -- clients -----------------------------------------------------------

    def client_for(self, ref: ModelRef, **kwargs: Any) -> OpenAI:
        """An OpenAI SDK client wired to ``ref``'s provider."""
        return make_client(ref.provider, **kwargs)

    def open(self, spec: str, **kwargs: Any) -> tuple[ModelRef, OpenAI]:
        """Resolve ``spec`` and return it alongside a ready client."""
        ref = self.resolve(spec)
        return ref, self.client_for(ref, **kwargs)


def make_client(provider: Provider, **kwargs: Any) -> OpenAI:
    """An OpenAI SDK client pointed at ``provider``."""
    from openai import OpenAI

    if provider.headers:
        kwargs.setdefault("default_headers", dict(provider.headers))
    return OpenAI(
        api_key=provider.require_api_key(),
        base_url=provider.base_url,
        **kwargs,
    )


def _prefix_hint(model_id: str) -> str | None:
    lowered = model_id.lower()
    matches = (p for p in _PREFIX_HINTS if lowered.startswith(p))
    best = max(matches, key=len, default=None)
    return _PREFIX_HINTS[best] if best is not None else None


def _read_config(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg = f"could not read config at {path}: {exc}"
        raise LLMError(msg) from exc


def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _provider_from_config(
    name: str, raw: Mapping[str, Any], existing: Provider | None
) -> Provider:
    if not isinstance(raw, dict):
        msg = f"[providers.{name}] must be a table"
        raise LLMError(msg)

    def pick(field_name: str, default: Any) -> Any:
        return raw.get(field_name, default)

    env_var = pick("env_var", existing.env_var if existing else None)
    base_url = pick("base_url", existing.base_url if existing else None)
    if not env_var or not base_url:
        msg = f"[providers.{name}] needs both env_var and base_url"
        raise LLMError(msg)

    return Provider(
        name=name,
        env_var=str(env_var),
        base_url=str(base_url),
        label=str(pick("label", existing.label if existing else "")),
        fallback_key=(
            str(raw["fallback_key"])
            if "fallback_key" in raw
            else (existing.fallback_key if existing else None)
        ),
        headers=dict(pick("headers", dict(existing.headers) if existing else {})),
    )


def iter_default_providers() -> Iterator[Provider]:
    """The built-in providers, in declaration order."""
    yield from _BUILTIN_PROVIDERS
