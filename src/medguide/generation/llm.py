"""Chat model factory over anthropic | openai | openrouter | ollama via LangChain's init_chat_model."""

from __future__ import annotations

import os
import re
from pathlib import Path

from medguide.config import OPENROUTER_BASE_URL, require_provider


def _no_think_ollama_class():
    """ChatOllama that appends a soft switch (e.g. Qwen3's "/no_think") to the system message.

    Ollama's `think=false` is ignored by some community GGUF chat templates, which then emit
    their reasoning as plain answer text. The prompt-level switch works regardless.
    """
    from langchain_ollama import ChatOllama

    class NoThinkChatOllama(ChatOllama):
        system_suffix: str = ""

        def _convert_messages_to_ollama_messages(self, messages):
            out = super()._convert_messages_to_ollama_messages(messages)
            if not self.system_suffix:
                return out
            if out and out[0].get("role") == "system":
                out[0] = {**out[0], "content": f"{out[0]['content']}\n{self.system_suffix}"}
            else:
                out.insert(0, {"role": "system", "content": self.system_suffix})
            return out

    return NoThinkChatOllama


def make_chat_model(
    provider: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    ollama: dict | None = None,
):
    """Build a LangChain chat model; raises ConfigError if the provider is not configured."""
    from langchain.chat_models import init_chat_model

    require_provider(provider)
    kwargs: dict = {"temperature": temperature}
    if provider == "ollama":
        opts = ollama or {}
        kwargs.update(
            num_predict=max_tokens,
            num_ctx=opts.get("num_ctx", 8192),
            reasoning=opts.get("reasoning", False),  # no hidden "thinking" tokens (speed)
            keep_alive=opts.get("keep_alive", "30m"),
            system_suffix="" if opts.get("reasoning") else opts.get("no_think_suffix", ""),
        )
        if os.getenv("OLLAMA_BASE_URL"):
            kwargs["base_url"] = os.environ["OLLAMA_BASE_URL"]
        return _no_think_ollama_class()(model=model, **kwargs)
    else:
        kwargs.update(max_tokens=max_tokens, max_retries=6)
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
    return make_chat_model(
        llm["provider"], llm["model"], llm["temperature"], llm["max_tokens"], cfg.get("ollama")
    )


def judge_from_config(cfg: dict, max_tokens: int = 2048):
    judge = cfg["judge"]
    return make_chat_model(judge["provider"], judge["model"], 0.0, max_tokens, cfg.get("ollama"))


def load_prompt(cfg: dict, name: str) -> str:
    return (Path(cfg["paths"]["prompts_dir"]) / name).read_text()


_THINK = re.compile(r"<think>.*?</think>|^\s*</think>", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove (possibly empty or half-open) reasoning blocks some local models emit."""
    return _THINK.sub("", text).strip()


def text_of(message) -> str:
    """Extract plain text from a LangChain AIMessage (content may be a list of blocks)."""
    content = message.content
    if not isinstance(content, str):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return strip_think(content)
