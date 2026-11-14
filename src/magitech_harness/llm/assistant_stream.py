"""Exact compact raw stream for one model attempt.

Each received chunk is recorded as exactly one timed record. Deltas
are never joined across boundaries: the harness has no delta-level
provider input, so run packing would fabricate boundaries the provider
never reported.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from magitech_harness.llm.types import ModelChunk


class AssistantStreamAccumulator:
    """Collects one attempt's exact compact raw stream, one chunk at a time."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def push(self, chunk: ModelChunk) -> None:
        """Append one received chunk as a single timed record."""
        self._records.append(
            {
                "type": "chunk",
                "time": int(time.time() * 1000),
                "chunk": chunk.model_dump(),
            }
        )

    def snapshot(self) -> list[dict[str, Any]]:
        """Detached copy of the records so far; the caller logs a snapshot."""
        return copy.deepcopy(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return len(self._records) > 0
