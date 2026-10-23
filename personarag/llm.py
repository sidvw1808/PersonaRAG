"""LLM backends behind one tiny interface: stream(system, messages) -> text chunks.

- ClaudeLLM: Anthropic API (best quality; needs ANTHROPIC_API_KEY).
- OllamaLLM: a local model served by Ollama (free, offline; quality depends on model size).
"""

import json
from collections.abc import Iterator
from typing import Protocol

import requests

from . import config


class LLM(Protocol):
    name: str

    def stream(self, system: str, messages: list[dict], max_tokens: int = 4000) -> Iterator[str]: ...


def complete(llm: LLM, system: str, messages: list[dict], max_tokens: int = 4000) -> str:
    return "".join(llm.stream(system, messages, max_tokens))


# Models that accept the server-side refusal fallback ("fallbacks": "default").
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class ClaudeLLM:
    def __init__(self, model: str = config.CLAUDE_MODEL, effort: str = config.CLAUDE_EFFORT):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.name = f"Claude ({model})"

    def stream(self, system: str, messages: list[dict], max_tokens: int = 4000) -> Iterator[str]:
        extra = {"output_config": {"effort": self.effort}}
        betas = []
        if config.CLAUDE_FALLBACKS and self.model in _FALLBACK_MODELS:
            extra["fallbacks"] = "default"
            betas.append("server-side-fallback-2026-07-01")

        with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            # The system prompt is identical across turns, so cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            betas=betas,
            extra_body=extra,
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n*[The model declined to answer this request.]*"
        elif final.stop_reason == "max_tokens":
            yield "\n\n*[Answer truncated: reached the token limit.]*"


class OllamaLLM:
    def __init__(self, model: str = config.OLLAMA_MODEL, url: str = config.OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")
        self.name = f"Ollama ({model})"

    def stream(self, system: str, messages: list[dict], max_tokens: int = 4000) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
            # Thinking models (gemma4, qwen3, deepseek-r1) otherwise spend the token budget
            # on hidden reasoning and can return an empty answer.
            "think": config.OLLAMA_THINK,
            "options": {"num_predict": max_tokens, "num_ctx": 8192},
        }
        with requests.post(f"{self.url}/api/chat", json=payload, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                if piece := data.get("message", {}).get("content"):
                    yield piece
                if data.get("done"):
                    break


def get_llm(provider: str | None = None) -> LLM:
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider == "claude":
        return ClaudeLLM()
    if provider == "ollama":
        return OllamaLLM()
    raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use 'claude' or 'ollama')")
