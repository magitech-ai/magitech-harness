"""Model-free tool-result pruning (``ctx.toolResultPruner``).

Ports dsh's ``@deepseek-ai/dsh-compaction-tool-result-pruner`` at
``packages/compaction/compaction-tool-result-pruner/src/index.ts``. Before a
broad conversation compaction runs, each over-budget ``tool/result`` on the
current surface is replaced by a head/middle/tail version whose removed span is
the fixed ``PRUNE_MARKER``. Each replacement cites the shadowed node and is
preceded by a ``compaction/prune`` event carrying that node's token estimate, so
a pure consumer can subtract it without per-node state.

Text is sliced by Unicode code point, so a retained boundary cannot split a
surrogate pair.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from animas import Context, Service

PRUNE_MARKER = "\n\n[... tool result middle pruned ...]\n\n"
DEFAULT_THRESHOLD_CHARS = 8192
DEFAULT_HEAD_CHARS = 4096
DEFAULT_TAIL_CHARS = 1024

_CONFIG_KEYS = frozenset({"thresholdChars", "headChars", "tailChars"})


class PruneConfigError(ValueError):
    """Raised when the pruning budgets are invalid."""


def code_point_length(text: str) -> int:
    """Count Unicode code points without splitting surrogate pairs."""
    return len(text)


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PruneConfigError(
            f"ToolResultPruneConfig: {name} ({value!r}) must be a positive integer"
        )
    return value


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PruneConfigError(
            f"ToolResultPruneConfig: {name} ({value!r}) must be a non-negative integer"
        )
    return value


def resolve_config(config: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Validate and resolve the pruning budgets (unknown keys fail loud)."""
    raw = dict(config or {})
    for key in raw:
        if key not in _CONFIG_KEYS:
            raise PruneConfigError(
                f'ToolResultPruneConfig: unknown key "{key}" '
                "(allowed: thresholdChars, headChars, tailChars)"
            )
    resolved = {
        "thresholdChars": _positive_int(
            "thresholdChars", raw.get("thresholdChars", DEFAULT_THRESHOLD_CHARS)
        ),
        "headChars": _non_negative_int(
            "headChars", raw.get("headChars", DEFAULT_HEAD_CHARS)
        ),
        "tailChars": _non_negative_int(
            "tailChars", raw.get("tailChars", DEFAULT_TAIL_CHARS)
        ),
    }
    emitted = (
        resolved["headChars"] + code_point_length(PRUNE_MARKER) + resolved["tailChars"]
    )
    if emitted > resolved["thresholdChars"]:
        raise PruneConfigError(
            "ToolResultPruneConfig: headChars + marker + tailChars "
            f"({emitted}) must be at most thresholdChars "
            f"({resolved['thresholdChars']})"
        )
    return resolved


def _text_blocks(blocks: Any) -> list[dict[str, Any]]:
    if not isinstance(blocks, list):
        return []
    return [
        block
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]


class ToolResultPruner(Service):
    """``ctx.toolResultPruner``: deterministic, replay-safe result pruning."""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(ctx, "toolResultPruner")
        self.config = resolve_config(config)

    def measure_content(self, blocks: Any) -> int:
        """Total Unicode code points across text blocks; others cost zero."""
        return sum(
            code_point_length(str(block.get("text", "")))
            for block in _text_blocks(blocks)
        )

    def prune_content(self, blocks: Any) -> list[dict[str, Any]] | None:
        """Replace an over-budget text middle, preserving rich-block order."""
        original = blocks if isinstance(blocks, list) else []
        total = self.measure_content(original)
        if total <= self.config["thresholdChars"]:
            return None
        removed_start = self.config["headChars"]
        removed_end = total - self.config["tailChars"]
        pruned: list[dict[str, Any]] = []
        consumed = 0
        marker_inserted = False
        for block in original:
            if not isinstance(block, dict) or block.get("type") != "text":
                pruned.append(block)
                continue
            points = list(str(block.get("text", "")))
            block_start = consumed
            block_end = block_start + len(points)
            head_end = min(len(points), max(0, removed_start - block_start))
            tail_start = min(len(points), max(0, removed_end - block_start))
            intersects = block_start < removed_end and block_end > removed_start
            marker = PRUNE_MARKER if intersects and not marker_inserted else ""
            if marker:
                marker_inserted = True
            text = "".join(points[:head_end]) + marker + "".join(points[tail_start:])
            if text:
                pruned.append({**block, "text": text})
            consumed = block_end
        if not marker_inserted:
            raise ValueError(
                "tool-result prune: failed to locate the removed text span"
            )
        after = self.measure_content(pruned)
        if after > self.config["thresholdChars"] or after >= total:
            raise ValueError(
                "tool-result prune: replacement must be smaller and within threshold"
            )
        return pruned

    def prune_session(self, session: Any) -> dict[str, Any]:
        """Rewrite every over-budget tool result on the current surface."""
        pruned: list[dict[str, Any]] = []
        chars_removed = 0
        for seq in list(session.surface_nodes()):
            event = session.event_at(seq)
            if event is None or event.type != "tool/result":
                continue
            message = event.payload.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            replacement = self.prune_content(content)
            if replacement is None:
                continue
            before = self.measure_content(content)
            after = self.measure_content(replacement)
            shadowed = _estimate_message(self.ctx, message)
            session.append(
                "compaction/prune",
                {
                    "shadowedRange": {"start": seq, "end": seq},
                    "shadowedSeqs": [seq],
                    "shadowedTokenCount": shadowed,
                },
            )
            new_message = {**message, "content": replacement}
            new_event = session.append(
                "tool/result",
                {**event.payload, "message": new_message},
                surface_op={"op": "replace", "startSeq": seq, "endSeq": seq},
                source_event_seqs=[seq],
            )
            pruned.append(
                {
                    "originalSeq": seq,
                    "replacementSeq": new_event.seq,
                    "callId": _call_id(message),
                    "charsBefore": before,
                    "charsAfter": after,
                }
            )
            chars_removed += before - after
        return {
            "pruned": bool(pruned),
            "entries": pruned,
            "charsRemoved": chars_removed,
        }


def _call_id(message: Mapping[str, Any]) -> str | None:
    source = message.get("source")
    if isinstance(source, Mapping):
        call_id = source.get("callId")
        if isinstance(call_id, str):
            return call_id
    call_id = message.get("toolCallId")
    return call_id if isinstance(call_id, str) else None


def _estimate_message(ctx: Context, message: Mapping[str, Any]) -> int:
    """Price one shadowed message through the token meter when available."""
    meter = ctx.get("tokenMeter", None, strict=False)
    if meter is not None and hasattr(meter, "estimate_message"):
        estimate = meter.estimate_message(dict(message))
        if isinstance(estimate, (int, float)) and not isinstance(estimate, bool):
            return int(estimate)
    # Heuristic fallback: four characters per token over the text content.
    chars = sum(
        code_point_length(str(block.get("text", "")))
        for block in _text_blocks(message.get("content"))
    )
    return math.ceil(chars / 4)


def apply(ctx: Context, config: dict[str, Any] | None = None) -> Any:
    """Mount the pruner service and register it with the compaction service."""
    pruner = ToolResultPruner(ctx, config)
    compaction = ctx.get("compaction", None, strict=False)
    if compaction is None:  # pragma: no cover - base bundle mounts compaction first
        return pruner

    def _run(session: Any, **_kwargs: Any) -> dict[str, Any]:
        return pruner.prune_session(session)

    disposer = compaction.register_pruner("tool-result", _run)
    return pruner, disposer


__all__ = [
    "DEFAULT_HEAD_CHARS",
    "DEFAULT_TAIL_CHARS",
    "DEFAULT_THRESHOLD_CHARS",
    "PRUNE_MARKER",
    "PruneConfigError",
    "ToolResultPruner",
    "apply",
    "code_point_length",
    "resolve_config",
]
