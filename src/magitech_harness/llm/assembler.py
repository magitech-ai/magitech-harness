"""Incremental stream-chunk assembler for model assistant turns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from magitech_harness.llm.brand import brand_string
from magitech_harness.llm.message import create_assistant_message
from magitech_harness.llm.types import (
    ContentBlock,
    FinishReason,
    ReplayEnvelope,
    StreamChunk,
    TokenUsage,
)
from magitech_harness.llm.values import assert_never


@dataclass
class _PartialBlock:
    block_type: str
    text: str = ""
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_arguments: str = ""
    block: ContentBlock | None = None


class BlockAssembler:
    """Incrementally assembles raw StreamChunks into ContentBlocks and messages."""

    def __init__(self) -> None:
        self._partials: dict[int, _PartialBlock] = {}
        self._order: list[int] = []
        self._usage: TokenUsage | None = None
        self._finish: FinishReason | None = None
        self._replay_state: ReplayEnvelope | None = None

    def push(self, chunk: StreamChunk) -> None:
        """Feed one raw chunk in stream order into the assembly state."""
        ctype = chunk.get("type")
        if ctype == "block-start":
            index = chunk["index"]
            if index not in self._partials:
                self._order.append(index)
                self._partials[index] = _PartialBlock(block_type=chunk["blockType"])
            return
        if ctype in ("text-delta", "reasoning-delta"):
            partial = self._ensure(
                chunk["index"], "text" if ctype == "text-delta" else "reasoning"
            )
            if partial.block is not None:
                return
            partial.text += chunk["text"]
            return
        if ctype == "tool-call-delta":
            partial = self._ensure(chunk["index"], "tool-call")
            if partial.block is not None:
                return
            partial.tool_call_id = chunk["id"]
            if chunk.get("name"):
                partial.tool_call_name = chunk["name"]
            partial.tool_call_arguments += chunk["argumentsDelta"]
            return
        if ctype == "block-end":
            block = chunk["block"]
            partial = self._ensure(chunk["index"], block["type"])
            if partial.block is not None:
                return
            partial.block = block
            return
        if ctype == "usage":
            self._usage = chunk["usage"]
            return
        if ctype == "finish":
            self._finish = chunk["reason"]
            self._replay_state = chunk.get("replayState")
            return
        assert_never(chunk, "BlockAssembler.push")

    def _ensure(self, index: int, block_type: str) -> _PartialBlock:
        partial = self._partials.get(index)
        if partial is None:
            partial = _PartialBlock(block_type=block_type)
            self._partials[index] = partial
            self._order.append(index)
        return partial

    def _assemble(self, partial: _PartialBlock, index: int) -> ContentBlock:
        if partial.block is not None:
            return partial.block
        if partial.block_type == "text":
            return {"type": "text", "text": partial.text}
        if partial.block_type == "reasoning":
            return {"type": "reasoning", "text": partial.text}
        if partial.block_type == "tool-call":
            return {
                "type": "tool-call",
                "id": partial.tool_call_id or brand_string(f"call-{index}"),
                "name": partial.tool_call_name or "",
                "arguments": partial.tool_call_arguments,
            }
        raise ValueError(
            f'cannot assemble incomplete block of type "{partial.block_type}"'
        )

    def _assembled(self) -> tuple[list[ContentBlock], ReplayEnvelope | None]:
        all_blocks = [
            self._assemble(self._partials[index], index) for index in self._order
        ]
        if self.finish.get("kind") == "max-tokens":
            kept = [block.get("type") != "tool-call" for block in all_blocks]
            blocks = [
                block for block, keep in zip(all_blocks, kept, strict=True) if keep
            ]
        else:
            kept = None
            blocks = all_blocks

        envelope = self._replay_state
        if envelope is None or "blocks" not in envelope or envelope["blocks"] is None:
            return blocks, envelope
        env_blocks = envelope["blocks"]
        if len(env_blocks) != len(all_blocks):
            return blocks, None
        if kept is None or len(blocks) == len(all_blocks):
            return blocks, envelope
        filtered_env: ReplayEnvelope = {
            "response": envelope["response"],
            "blocks": [
                meta for meta, keep in zip(env_blocks, kept, strict=True) if keep
            ],
        }
        return blocks, filtered_env

    def blocks(self) -> list[ContentBlock]:
        """Assemble all blocks seen so far in stream order."""
        return self._assembled()[0]

    def interrupted_blocks(self) -> list[ContentBlock]:
        """Assemble non-whitespace text and reasoning blocks for an interrupted turn."""
        kept: list[ContentBlock] = []
        for index in self._order:
            partial = self._partials[index]
            btype = (
                partial.block.get("type")
                if partial.block is not None
                else partial.block_type
            )
            if btype not in ("text", "reasoning"):
                continue
            block = self._assemble(partial, index)
            text = block.get("text")
            if isinstance(text, str) and text.strip() != "":
                kept.append(block)
        return kept

    @property
    def usage(self) -> TokenUsage | None:
        """Usage from the last usage chunk, or None before one arrives."""
        return self._usage

    @property
    def finish(self) -> FinishReason:
        """Finish reason from the last finish chunk, defaulting to stop."""
        return self._finish if self._finish is not None else {"kind": "stop"}

    @property
    def replay_state(self) -> ReplayEnvelope | None:
        """Replay metadata aligned with blocks(), or None if lengths disagree."""
        return self._assembled()[1]

    @property
    def replayState(self) -> ReplayEnvelope | None:
        """CamelCase alias for replay_state."""
        return self.replay_state

    def message(self, source: Mapping[str, Any]) -> dict[str, Any]:
        """Assemble a frozen assistant message carrying the caller's attribution."""
        return create_assistant_message({"content": self.blocks(), "source": source})

    interruptedBlocks = interrupted_blocks
