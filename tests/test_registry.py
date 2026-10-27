from __future__ import annotations

import pytest

from llm.registry import (
    LLMError,
    MissingAPIKeyError,
    Registry,
    UnknownModelError,
)


@pytest.fixture
def bare(tmp_path) -> Registry:
    """A registry with no user config layered on top."""
    return Registry.load(tmp_path / "missing.toml")


@pytest.mark.parametrize(
    ("spec", "provider", "model_id"),
    [
        ("kimi-k3", "moonshot", "kimi-k3"),
        ("moonshot/kimi-k3", "moonshot", "kimi-k3"),
        ("gpt-5", "openai", "gpt-5"),
        ("claude-opus-5", "anthropic", "claude-opus-5"),
        ("deepseek-reasoner", "deepseek", "deepseek-reasoner"),
        ("grok-4", "xai", "grok-4"),
        ("glm-4.6", "zai", "glm-4.6"),
        ("gemini-3-pro", "google", "gemini-3-pro"),
        ("  gpt-5  ", "openai", "gpt-5"),
        ("openrouter/anthropic/claude-opus-5", "openrouter", "anthropic/claude-opus-5"),
    ],
)
def test_resolve(bare: Registry, spec: str, provider: str, model_id: str) -> None:
    ref = bare.resolve(spec)
    assert ref.provider.name == provider
    assert ref.model_id == model_id


def test_unknown_model_is_rejected(bare: Registry) -> None:
    with pytest.raises(UnknownModelError, match="cannot tell which provider"):
        bare.resolve("some-model-nobody-knows")


def test_unknown_provider_prefix_is_rejected(bare: Registry) -> None:
    with pytest.raises(UnknownModelError):
        bare.resolve("nosuchprovider/whatever")


def test_empty_spec_is_rejected(bare: Registry) -> None:
    with pytest.raises(UnknownModelError):
        bare.resolve("   ")


def test_prefix_match_prefers_the_longest_hint(bare: Registry) -> None:
    # "mistral" and "ministral" both exist; neither should shadow the other.
    assert bare.resolve("ministral-8b").provider.name == "mistral"


def test_config_aliases_and_default(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'default_model = "kimi"\n[aliases]\nkimi = "moonshot/kimi-k3"\n',
        encoding="utf-8",
    )
    registry = Registry.load(path)
    assert registry.default_model == "kimi"
    assert registry.resolve("kimi").spec == "moonshot/kimi-k3"


def test_alias_cycles_do_not_hang(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[aliases]\na = "b"\nb = "a"\n', encoding="utf-8")
    with pytest.raises(UnknownModelError):
        Registry.load(path).resolve("a")


def test_config_can_add_a_provider(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[providers.internal]\n"
        'env_var = "INTERNAL_KEY"\n'
        'base_url = "https://llm.internal/v1"\n',
        encoding="utf-8",
    )
    ref = Registry.load(path).resolve("internal/my-model")
    assert ref.provider.env_var == "INTERNAL_KEY"
    assert ref.provider.base_url == "https://llm.internal/v1"


def test_config_can_override_a_builtin_base_url(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[providers.moonshot]\nbase_url = "https://api.moonshot.cn/v1"\n',
        encoding="utf-8",
    )
    ref = Registry.load(path).resolve("kimi-k3")
    assert ref.provider.base_url == "https://api.moonshot.cn/v1"
    assert ref.provider.env_var == "MOONSHOT_API_KEY"  # inherited
    assert ref.reasoning_replay == "all"  # inherited too


@pytest.mark.parametrize(
    ("spec", "replay"),
    [
        ("kimi-k3", "all"),
        ("kimi-k3.1-pro", "all"),
        ("kimi-k2.7-code-highspeed", "all"),
        ("kimi-k2.6", "turn"),
        ("gpt-5", "none"),
        ("deepseek-reasoner", "none"),
    ],
)
def test_reasoning_replay_follows_the_model(
    bare: Registry, spec: str, replay: str
) -> None:
    assert bare.resolve(spec).reasoning_replay == replay


def test_incomplete_custom_provider_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[providers.broken]\nlabel = "nope"\n', encoding="utf-8")
    with pytest.raises(LLMError, match="env_var and base_url"):
        Registry.load(path)


def test_api_key_comes_from_the_environment(bare: Registry, monkeypatch) -> None:
    provider = bare.provider("moonshot")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    assert not provider.is_ready()
    with pytest.raises(MissingAPIKeyError, match="MOONSHOT_API_KEY"):
        provider.require_api_key()

    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    assert provider.is_ready()
    assert provider.require_api_key() == "sk-test"


def test_blank_env_var_is_not_a_key(bare: Registry, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    assert not bare.provider("groq").is_ready()


def test_local_providers_need_no_key(bare: Registry, monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert bare.provider("ollama").is_ready()


def test_client_targets_the_provider(bare: Registry, monkeypatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    ref, client = bare.open("kimi-k3")
    assert ref.model_id == "kimi-k3"
    assert str(client.base_url).rstrip("/") == "https://api.moonshot.ai/v1"
    assert client.api_key == "sk-test"


def test_ready_providers_reflects_the_environment(bare: Registry, monkeypatch) -> None:
    for provider in bare.sorted_providers():
        monkeypatch.delenv(provider.env_var, raising=False)
    monkeypatch.setenv("XAI_API_KEY", "sk-test")
    names = {p.name for p in bare.ready_providers()}
    assert "xai" in names
    assert "openai" not in names
