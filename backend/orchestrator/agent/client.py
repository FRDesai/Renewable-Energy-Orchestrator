"""Thin wrapper over the Anthropic SDK.

Everything in this package must work with no API key configured: the PRD requires
that "if the LLM is unavailable, the optimizer runs the loop alone on schedule with
template explanations". `available` is the single switch the rest of the code checks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

MODEL = "claude-sonnet-5"
MAX_TOKENS = 16_000


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "errors": self.errors}


class LLMClient:
    def __init__(self, *, model: str = MODEL, effort: str = "medium",
                 api_key: str | None = None) -> None:
        self.model = model
        self.effort = effort
        self.usage = Usage()
        self._client: Any | None = None
        self._reason: str = ""
        try:
            import anthropic
        except ImportError:                       # pragma: no cover
            self._reason = "anthropic SDK not installed"
            return
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if not key and not token:
            self._reason = "no ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN configured"
            return
        try:
            self._client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        except Exception as exc:                  # pragma: no cover
            self._reason = f"client construction failed: {exc}"

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def unavailable_reason(self) -> str:
        return self._reason

    def messages(self, **kwargs) -> Any:
        """One non-streaming request. Raises on failure; the caller handles retry."""
        if self._client is None:
            raise RuntimeError(f"LLM unavailable: {self._reason}")
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("max_tokens", MAX_TOKENS)
        kwargs.setdefault("output_config", {"effort": self.effort})
        self.usage.calls += 1
        try:
            response = self._client.messages.create(**kwargs)
        except Exception:
            self.usage.errors += 1
            raise
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.usage.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return response

    def parse(self, *, output_format, **kwargs) -> Any:
        """Structured output against a Pydantic model."""
        if self._client is None:
            raise RuntimeError(f"LLM unavailable: {self._reason}")
        kwargs.setdefault("model", self.model)
        kwargs.setdefault("max_tokens", MAX_TOKENS)
        self.usage.calls += 1
        try:
            response = self._client.messages.parse(output_format=output_format, **kwargs)
        except Exception:
            self.usage.errors += 1
            raise
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.usage.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return response

    @staticmethod
    def text_of(response: Any) -> str:
        parts = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()
