"""Ollama wrapper used by every agent.

One chokepoint for LLM calls so we can swap providers or add retry/logging
without touching agent code.
"""
from __future__ import annotations
import json
from typing import Any
import ollama
from config import OLLAMA_HOST, MODEL


_client = ollama.Client(host=OLLAMA_HOST)


def chat(
    system: str,
    user: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> str:
    """Single-turn chat. json_mode=True forces Ollama to emit valid JSON."""
    options = {"num_predict": max_tokens, "temperature": temperature}
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": options,
    }
    if json_mode:
        kwargs["format"] = "json"
    return _client.chat(**kwargs)["message"]["content"]


def chat_json(system: str, user: str, **kwargs) -> dict:
    """Chat in JSON mode and parse the result."""
    raw = chat(system, user, json_mode=True, **kwargs)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {raw!r}") from e
