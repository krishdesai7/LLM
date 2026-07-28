"""Chat with any OpenAI-compatible model from one place.

The registry maps a model name to the provider that serves it, so naming
``kimi-k3`` is enough to infer ``MOONSHOT_API_KEY`` and
``https://api.moonshot.ai/v1``.

    >>> from llm import Registry
    >>> registry = Registry.load()
    >>> ref, client = registry.open("kimi-k3")  # doctest: +SKIP
    >>> client.chat.completions.create(  # doctest: +SKIP
    ...     model=ref.model_id, messages=[{"role": "user", "content": "hi"}]
    ... )
"""

from .__main__ import main
from .chat import ChatSession, Message
from .registry import (
    MissingAPIKeyError,
    ModelRef,
    Provider,
    Registry,
    UnknownModelError,
    make_client,
)

__all__: list[str] = [
    "ChatSession",
    "Message",
    "MissingAPIKeyError",
    "ModelRef",
    "Provider",
    "Registry",
    "UnknownModelError",
    "main",
    "make_client",
]
