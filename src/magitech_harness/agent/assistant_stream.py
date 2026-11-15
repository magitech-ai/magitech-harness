"""Process-local assistant attempt framing and durable stream accumulation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

from magitech_harness.agent.contracts import AssistantStreamFrame
from magitech_harness.llm.assembler import BlockAssembler
from magitech_harness.llm.assistant_stream import AssistantStreamAccumulator
from magitech_harness.llm.types import (
    ContentBlock,
    FinishReason,
    ReplayEnvelope,
    StreamChunk,
    TokenUsage,
)


class AssistantStreamAttempt:
    """Folds one model attempt into one compact stream plus ordered transient frames."""

    def __init__(
        self,
        session_id: str,
        attempt: int,
        next_revision: Callable[[], int],
        turn: int,
        step: int,
        emit: Callable[[AssistantStreamFrame], None],
    ) -> None:
        self.attempt_id = f"{session_id}:{attempt}"
        self.turn = turn
        self.step = step
        self._next_revision = next_revision
        self._emit = emit
        self._accumulator = AssistantStreamAccumulator()
        self._assembler = BlockAssembler()
        self._index = 0
        self._terminal = False

    @property
    def attemptId(self) -> str:
        """CamelCase alias matching the frame field name."""
        return self.attempt_id

    @property
    def ended(self) -> bool:
        """Whether this started attempt has emitted its terminal frame."""
        return self._terminal

    def start(self) -> None:
        """Publish the opening marker before the first delivered chunk."""
        self._emit(
            {
                "type": "start",
                "attemptId": self.attempt_id,
                "revision": self._next_revision(),
                "turn": self.turn,
                "step": self.step,
            }
        )

    def push(self, chunk: StreamChunk | Mapping[str, Any]) -> None:
        """Snapshot one chunk once, then feed compaction, assembly, and live frames."""
        timed = self._accumulator.push(
            {"time": int(time.time() * 1000), "chunk": chunk}
        )
        assert timed is not None
        self._assembler.push(timed["chunk"])
        index = self._index
        self._index += 1
        self._emit(
            {
                "type": "chunk",
                "attemptId": self.attempt_id,
                "revision": self._next_revision(),
                "index": index,
                "time": timed["time"],
                "chunk": timed["chunk"],
            }
        )

    def settle(
        self,
        event_type: Literal["assistant/message", "assistant/attempt"],
        append: Callable[[], int],
    ) -> None:
        """Publish terminal settlement after the matching durable event commits."""
        try:
            seq = append()
        except Exception:
            self.abandon()
            raise
        self._terminal = True
        self._emit(
            {
                "type": "end",
                "attemptId": self.attempt_id,
                "revision": self._next_revision(),
                "index": self._index,
                "outcome": {
                    "kind": "committed",
                    "eventType": event_type,
                    "seq": seq,
                },
            }
        )

    def abandon(self) -> None:
        """Publish abandonment when no durable attempt event can be committed."""
        self._terminal = True
        self._emit(
            {
                "type": "end",
                "attemptId": self.attempt_id,
                "revision": self._next_revision(),
                "index": self._index,
                "outcome": {"kind": "abandoned"},
            }
        )

    @property
    def stream(self) -> list[dict[str, Any]]:
        """Exact compact stream for the final durable event."""
        return list(self._accumulator.snapshot())

    def blocks(self) -> list[ContentBlock]:
        """Canonical completed-message blocks from the same chunks."""
        return self._assembler.blocks()

    def interrupted_blocks(self) -> list[ContentBlock]:
        """Safe visible prefix when cancellation interrupts the attempt."""
        return self._assembler.interrupted_blocks()

    interruptedBlocks = interrupted_blocks

    @property
    def usage(self) -> TokenUsage | None:
        """Latest adapter-reported usage in the stream."""
        return self._assembler.usage

    @property
    def finish(self) -> FinishReason:
        """Terminal reason, defaulting to stop when the stream omitted one."""
        return self._assembler.finish

    @property
    def replay_state(self) -> ReplayEnvelope | None:
        """Replay state carried by the terminal finish record."""
        return self._assembler.replay_state

    @property
    def replayState(self) -> ReplayEnvelope | None:
        """CamelCase alias for ``replay_state``."""
        return self._assembler.replay_state


__all__ = ["AssistantStreamAttempt"]
