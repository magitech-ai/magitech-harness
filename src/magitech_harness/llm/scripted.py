"""Config-driven scripted LLM adapter for offline end-to-end testing.

Drives the unified agent loop with a fixed sequence of raw stream chunks,
loaded from a JSON script. Each call to ``stream()`` consumes one turn, so a
tool-call turn followed by a final-text turn reproduces a full tool loop from
an installed wheel. Fail loud when the script is exhausted — an under-scripted
turn is a test failure, never a silent hang.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.service import LlmAdapter
from magitech_harness.llm.types import GenerateOptions, LlmModelInfo, StreamChunk

DEFAULT_PROVIDER = "scripted"
DEFAULT_MODEL = "script"


@dataclass(frozen=True)
class ScriptSpec:
    provider: str
    model: str
    turns: tuple[tuple[dict[str, Any], ...], ...]


def load_script(path: Path | str) -> ScriptSpec:
    """Parse and validate a script file into a ScriptSpec."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"script is not valid JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("script must be a JSON object")  # noqa: TRY004
    provider = data.get("provider", DEFAULT_PROVIDER)
    model = data.get("model", DEFAULT_MODEL)
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("script provider must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("script model must be a non-empty string")
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError("script turns must be a non-empty list")
    turns: list[tuple[dict[str, Any], ...]] = []
    for index, turn in enumerate(raw_turns):
        if not isinstance(turn, list):
            raise ValueError(  # noqa: TRY004
                f"script turn {index} must be a list of chunks"
            )
        chunks: list[dict[str, Any]] = []
        for chunk in turn:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("type"), str):
                raise ValueError(  # noqa: TRY004
                    f"script turn {index} has a chunk without a string type"
                )
            chunks.append(dict(chunk))
        turns.append(tuple(chunks))
    return ScriptSpec(provider=provider, model=model, turns=tuple(turns))


class ScriptedLlmAdapter(LlmAdapter):
    """Streams one scripted turn per ``stream()`` call."""

    def __init__(self, spec: ScriptSpec) -> None:
        self._spec = spec
        self._cursor = 0

    @property
    def provider(self) -> str:
        return self._spec.provider

    @property
    def model(self) -> str:
        return self._spec.model

    async def list_models(self, provider: str) -> Sequence[LlmModelInfo]:
        del provider
        return [
            {
                "provider": self.provider,
                "id": self.model,
                "name": self.model,
            }
        ]

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        del options
        if self._cursor >= len(self._spec.turns):
            raise LlmError(
                f"scripted adapter exhausted after {len(self._spec.turns)} turn(s)",
                "SCRIPT_EXHAUSTED",
            )
        turn = self._spec.turns[self._cursor]
        self._cursor += 1
        for chunk in turn:
            yield dict(chunk)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "ScriptSpec",
    "ScriptedLlmAdapter",
    "load_script",
]
