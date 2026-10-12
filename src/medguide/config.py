"""Config loading, env overrides and hardware detection."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"

PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
}
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ConfigError(RuntimeError):
    """Raised when the configuration is incomplete or invalid."""


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply_env(cfg: dict) -> dict:
    """Env vars win over YAML for provider/model selection and Qdrant URL."""
    env_map = {
        ("llm", "provider"): "LLM_PROVIDER",
        ("llm", "model"): "LLM_MODEL",
        ("judge", "provider"): "JUDGE_PROVIDER",
        ("judge", "model"): "JUDGE_MODEL",
        ("retrieval", "mode"): "RETRIEVAL_MODE",
        ("vector_store", "mode"): "VECTOR_STORE_MODE",
    }
    for (section, key), env in env_map.items():
        if os.getenv(env):
            cfg[section][key] = os.environ[env]
    cfg["vector_store"]["url"] = os.getenv("QDRANT_URL", "http://localhost:6333")
    return cfg


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load YAML config, then apply env vars and explicit overrides."""
    load_dotenv(ROOT / ".env", override=False)
    path = Path(path or os.getenv("GUIDECITE_CONFIG", DEFAULT_CONFIG))
    if not path.is_absolute():
        path = ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = _apply_env(cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str(ROOT / rel)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    return load_config()


def require_provider(provider: str) -> None:
    """Fail early with a readable message if a provider is unknown or has no API key."""
    if provider not in PROVIDER_KEYS:
        raise ConfigError(
            f"Unknown LLM provider '{provider}'. Choose one of: {', '.join(PROVIDER_KEYS)}."
        )
    key = PROVIDER_KEYS[provider]
    if key and not os.getenv(key):
        raise ConfigError(
            f"Provider '{provider}' selected but {key} is not set. "
            f"Add it to .env (see .env.example) or choose another provider via LLM_PROVIDER."
        )


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embedding_settings(cfg: dict) -> tuple[str, str, int]:
    """Return (model_name, device, batch_size) adapted to the available hardware."""
    device = detect_device()
    emb = cfg["embedding"]
    model = emb["model"] if device != "cpu" else emb["cpu_fallback_model"]
    batch = emb.get("batch_size") or {"cuda": 128, "mps": 64, "cpu": 16}[device]
    return model, device, batch
