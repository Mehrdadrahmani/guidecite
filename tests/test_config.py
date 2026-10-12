import pytest

from medguide.config import ConfigError, load_config, require_provider


def test_defaults_load():
    cfg = load_config()
    assert cfg["language"] == "en"
    assert cfg["retrieval"]["mode"] in {"dense", "bm25", "hybrid", "hybrid_rerank"}
    assert cfg["paths"]["prompts_dir"].endswith("prompts")


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    cfg = load_config()
    assert cfg["llm"]["provider"] == "ollama"
    assert cfg["llm"]["model"] == "llama3.1:8b"
    assert cfg["retrieval"]["mode"] == "bm25"


def test_explicit_overrides_merge():
    cfg = load_config(overrides={"retrieval": {"top_k": 7}})
    assert cfg["retrieval"]["top_k"] == 7
    assert cfg["retrieval"]["rrf_k"] == 60


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        require_provider("anthropic")
    with pytest.raises(ConfigError, match="Unknown LLM provider"):
        require_provider("gemini")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        require_provider("openrouter")
    require_provider("ollama")  # no key needed
