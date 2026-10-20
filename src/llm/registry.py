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
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from openai import OpenAI

__all__: list[str] = [
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
        key: str = os.environ.get(key=self.env_var, default="").strip()
        return key or self.fallback_key

    def is_ready(self) -> bool:
        return self.api_key() is not None

    def require_api_key(self) -> str:
        key: str | None = self.api_key()
        if key is None:
            msg: str = (
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
    Provider(
        name="openai",
        env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        label="OpenAI",
    ),
    Provider(
        name="anthropic",
        env_var="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        label="Anthropic",
    ),
    Provider(
        name="google",
        env_var="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        label="Google Gemini",
    ),
    Provider(
        name="moonshot",
        env_var="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        label="Moonshot (Kimi)",
    ),
    Provider(
        name="deepseek",
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        label="DeepSeek",
    ),
    Provider(
        name="xai",
        env_var="XAI_API_KEY",
        base_url="https://api.x.ai/v1",
        label="xAI (Grok)",
    ),
    Provider(
        name="zai",
        env_var="ZAI_API_KEY",
        base_url="https://api.z.ai/api/paas/v4",
        label="Z.ai (GLM)",
    ),
    Provider(
        name="mistral",
        env_var="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        label="Mistral",
    ),
    Provider(
        name="groq",
        env_var="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        label="Groq",
    ),
    Provider(
        name="cerebras",
        env_var="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        label="Cerebras",
    ),
    Provider(
        name="together",
        env_var="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        label="Together",
    ),
    Provider(
        name="fireworks",
        env_var="FIREWORKS_API_KEY",
        base_url="https://api.fireworks.ai/inference/v1",
        label="Fireworks",
    ),
    Provider(
        name="deepinfra",
        env_var="DEEPINFRA_API_KEY",
        base_url="https://api.deepinfra.com/v1/openai",
        label="DeepInfra",
    ),
    Provider(
        name="perplexity",
        env_var="PERPLEXITY_API_KEY",
        base_url="https://api.perplexity.ai",
        label="Perplexity",
    ),
    Provider(
        name="openrouter",
        env_var="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        label="OpenRouter",
    ),
    Provider(
        name="ollama",
        env_var="OLLAMA_API_KEY",
        base_url="http://localhost:11434/v1",
        label="Ollama (local)",
        fallback_key="ollama",
    ),
    Provider(
        name="lmstudio",
        env_var="LMSTUDIO_API_KEY",
        base_url="http://localhost:1234/v1",
        label="LM Studio (local)",
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
    override: str | None = os.environ.get(key="LLM_CONFIG")
    if override:
        return Path(override).expanduser()
    base: str | None = os.environ.get(key="XDG_CONFIG_HOME")
    root: Path = Path(base).expanduser() if base else Path.home() / ".config"
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
        providers: dict[str, Provider] = {p.name: p for p in _BUILTIN_PROVIDERS}
        data: Mapping[str, Any] = _read_config(
            path=path if path is not None else config_path()
        )

        for name, raw in _table(data, key="providers").items():
            providers[name] = _provider_from_config(
                name, raw, existing=providers.get(name)
            )

        aliases: dict[str, str] = {
            str(object=k): str(object=v)
            for k, v in _table(data, key="aliases").items()
            if str(object=v).strip()
        }
        default: Any | None = data.get("default_model")
        return cls(
            providers=providers,
            aliases=aliases,
            default_model=str(object=default) if default else None,
        )

    # -- lookup ------------------------------------------------------------

    def provider(self, name: str) -> Provider:
        try:
            return self.providers[name]
        except KeyError:
            known: str = ", ".join(sorted(self.providers))
            msg: str = f"unknown provider {name!r}; known providers: {known}"
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
        name: str = spec.strip()
        if not name:
            msg = "no model given"
            raise UnknownModelError(msg)

        for _ in range(len(self.aliases) + 1):
            target: str | None = self.aliases.get(name)
            if target is None or target == name:
                break
            name: str = target.strip()

        if "/" in name:
            head, _, tail = name.partition("/")
            if head in self.providers and tail:
                return ModelRef(provider=self.providers[head], model_id=tail)

        hint: str | None = _prefix_hint(model_id=name)
        if hint is not None and hint in self.providers:
            return ModelRef(provider=self.providers[hint], model_id=name)

        msg: str = (
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
        ref: ModelRef = self.resolve(spec)
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
    lowered: str = model_id.lower()
    matches: Generator[str] = (p for p in _PREFIX_HINTS if lowered.startswith(p))
    best: str | None = max(matches, key=len, default=None)
    return _PREFIX_HINTS[best] if best is not None else None


def _read_config(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        msg: str = f"could not read config at {path}: {exc}"
        raise LLMError(msg) from exc


def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value: Any | None = data.get(key)
    return value if isinstance(value, dict) else {}


def _provider_from_config(
    name: str, raw: Mapping[str, Any], existing: Provider | None
) -> Provider:
    if not isinstance(raw, dict):
        msg: str = f"[providers.{name}] must be a table"
        raise LLMError(msg)

    def pick(field_name: str, default: Any) -> Any:
        return raw.get(field_name, default)

    env_var: Any = pick(
        field_name="env_var", default=existing.env_var if existing else None
    )
    base_url: Any = pick(
        field_name="base_url", default=existing.base_url if existing else None
    )
    if not env_var or not base_url:
        msg = f"[providers.{name}] needs both env_var and base_url"
        raise LLMError(msg)

    return Provider(
        name=name,
        env_var=str(object=env_var),
        base_url=str(object=base_url),
        label=str(
            object=pick(field_name="label", default=existing.label if existing else "")
        ),
        fallback_key=(
            str(object=raw["fallback_key"])
            if "fallback_key" in raw
            else (existing.fallback_key if existing else None)
        ),
        headers=dict(
            pick(
                field_name="headers", default=dict(existing.headers) if existing else {}
            )
        ),
    )


def iter_default_providers() -> Iterator[Provider]:
    """The built-in providers, in declaration order."""
    yield from _BUILTIN_PROVIDERS
