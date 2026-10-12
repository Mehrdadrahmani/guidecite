"""Chat model factory over anthropic | openai | openrouter | ollama via LangChain's init_chat_model."""

from __future__ import annotations

import os
from pathlib import Path

from medguide.config import OPENROUTER_BASE_URL, require_provider


def make_chat_model(provider: str, model: str, temperature: float = 0.0, max_tokens: int = 1024):
    """Build a LangChain chat model; raises ConfigError if the provider is not configured."""
    from langchain.chat_models import init_chat_model

    require_provider(provider)
    kwargs: dict = {"temperature": temperature, "max_tokens": max_tokens, "max_retries": 6}
    if provider == "openrouter":
        # OpenRouter speaks the OpenAI API; model ids look like "anthropic/claude-haiku-5-5"
        kwargs.update(
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        provider = "openai"
    return init_chat_model(model, model_provider=provider, **kwargs)


def generator_from_config(cfg: dict):
    llm = cfg["llm"]
    return make_chat_model(llm["provider"], llm["model"], llm["temperature"], llm["max_tokens"])


def judge_from_config(cfg: dict, max_tokens: int = 2048):
    judge = cfg["judge"]
    return make_chat_model(judge["provider"], judge["model"], 0.0, max_tokens)


def load_prompt(cfg: dict, name: str) -> str:
    return (Path(cfg["paths"]["prompts_dir"]) / name).read_text()


def text_of(message) -> str:
    """Extract plain text from a LangChain AIMessage (content may be a list of blocks)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))
