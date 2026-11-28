"""Reference-shaped NDJSON projection for ``mth headless --json``.

Every emitted item is derived from a committed mth session event, so the
projection cannot put model-visible text into the stream without a matching
entry in the append-only log. The wire vocabulary (`session`, `status`,
`thinking`, `text`, `tool_call`, `tool_result`, `final`) and the byte bounds
follow the pinned dsh headless runner; brand strings stay mth-owned.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any, cast

MAX_STRING_BYTES = 8 * 1024
MAX_EVENT_BYTES = 32 * 1024
LINE_TERMINATOR_BYTES = 1
MAX_DEPTH = 64


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Cut a string at a UTF-8 byte boundary, dropping a split trailing character."""
    cut = text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
    return cut


def _bound_value(
    value: Any, max_bytes: int, state: dict[str, bool], depth: int = 0
) -> Any:
    if isinstance(value, str):
        if len(value.encode("utf-8")) <= max_bytes:
            return value
        state["truncated"] = True
        return truncate_utf8(value, max_bytes)
    if isinstance(value, list):
        if depth >= MAX_DEPTH:
            state["truncated"] = True
            return "[truncated: depth]"
        return [_bound_value(item, max_bytes, state, depth + 1) for item in value]
    if isinstance(value, dict):
        if depth >= MAX_DEPTH:
            state["truncated"] = True
            return "[truncated: depth]"
        bounded: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and len(key.encode("utf-8")) > max_bytes:
                state["truncated"] = True
                key = truncate_utf8(key, max_bytes)
            bounded[key] = _bound_value(item, max_bytes, state, depth + 1)
        return bounded
    return value


def bound_json_event(
    event: dict[str, Any], max_string_bytes: int = MAX_STRING_BYTES
) -> dict[str, Any]:
    """Cap every string and key, flagging the payload when anything was cut."""
    state = {"truncated": False}
    bounded = cast(dict[str, Any], _bound_value(event, max_string_bytes, state))
    if state["truncated"]:
        bounded["truncated"] = True
    return bounded


def bound_json_line(
    event: dict[str, Any],
    max_string_bytes: int = MAX_STRING_BYTES,
    max_event_bytes: int = MAX_EVENT_BYTES,
) -> str:
    """Serialize one event under both the per-string and the whole-line cap."""
    limit = max_event_bytes - LINE_TERMINATOR_BYTES
    bounded = bound_json_event(event, max_string_bytes)
    line = json.dumps(bounded, ensure_ascii=False, separators=(",", ":"))
    if len(line.encode("utf-8")) <= limit:
        return line
    scalars = {
        key: value
        for key, value in bounded.items()
        if value is None or not isinstance(value, (dict, list))
    }
    scalars["truncated"] = True
    short = json.dumps(scalars, ensure_ascii=False, separators=(",", ":"))
    if len(short.encode("utf-8")) <= limit:
        return short
    return json.dumps(
        {"type": bounded.get("type"), "truncated": True}, separators=(",", ":")
    )


def _reject_constant(value: str) -> Any:
    raise ValueError(value)


def parse_arguments(raw: Any) -> Any:
    """Parse tool-call arguments as JSON, keeping non-JSON text verbatim."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return raw
    if raw == "":
        return {}
    try:
        return json.loads(raw, parse_constant=_reject_constant)
    except (ValueError, TypeError):
        return raw


def result_text(content: Any) -> str:
    """Join the text blocks of a committed tool result's content."""
    if not isinstance(content, list):
        return "" if content is None else str(content)
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def add_usage(
    total: dict[str, Any] | None, sample: Any
) -> tuple[dict[str, Any] | None, bool]:
    """Accumulate numeric usage fields; a non-numeric sample makes the total incomplete."""
    if sample is None:
        return total, total is not None
    if not isinstance(sample, dict):
        return total, False
    merged = dict(total or {})
    for key, value in sample.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            return total, False
        merged[key] = merged.get(key, 0) + value
    return merged, True


class HeadlessJsonProjection:
    """Project one agent's committed session events as NDJSON on a sink."""

    def __init__(self, sink: Callable[[str], None], session_id: str, cwd: str) -> None:
        self._sink = sink
        self._session_id = session_id
        self._usage: dict[str, Any] | None = None
        self._usage_complete = True
        self.write({"type": "session", "sessionId": session_id, "cwd": cwd})

    def write(self, event: dict[str, Any]) -> None:
        self._sink(bound_json_line(event) + "\n")

    def on_event(
        self, event_type: str, payload: dict[str, Any], surface_op: Any = None
    ) -> None:
        """Map one committed session event to its reference-shaped wire event."""
        if event_type == "turn/start":
            self.write(
                {"type": "status", "phase": "turn_start", "turn": payload.get("turn")}
            )
        elif event_type == "step/start":
            self.write(
                {
                    "type": "status",
                    "phase": "step_start",
                    "turn": payload.get("turn"),
                    "step": payload.get("step"),
                }
            )
        elif event_type == "assistant/attempt":
            self._usage, self._usage_complete = add_usage(
                self._usage, payload.get("stream")
            )
        elif event_type == "assistant/message":
            self._usage, self._usage_complete = add_usage(
                self._usage, payload.get("usage", payload.get("stream"))
            )
            message = payload.get("message")
            blocks = message.get("content", []) if isinstance(message, dict) else []
            for block in blocks if isinstance(blocks, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "reasoning" and isinstance(
                    block.get("text"), str
                ):
                    self.write({"type": "thinking", "text": block["text"]})
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    self.write({"type": "text", "text": block["text"]})
        elif event_type == "step/end":
            usage, complete = self._usage, self._usage_complete
            self._usage, self._usage_complete = None, True
            status: dict[str, Any] = {
                "type": "status",
                "phase": "step_end",
                "turn": payload.get("turn"),
                "step": payload.get("step"),
            }
            if complete and usage is not None:
                status["usage"] = usage
            self.write(status)
        elif event_type == "turn/end":
            self.write(
                {
                    "type": "status",
                    "phase": "turn_end",
                    "turn": payload.get("turn"),
                    "reason": payload.get("reason"),
                }
            )
        elif event_type == "tool/call":
            self.write(
                {
                    "type": "tool_call",
                    "callId": payload.get("callId"),
                    "tool": payload.get("name"),
                    "input": parse_arguments(payload.get("arguments")),
                }
            )
        elif event_type == "tool/result" and surface_op == "append":
            message = payload.get("message")
            if not isinstance(message, dict):
                return
            result: dict[str, Any] = {
                "type": "tool_result",
                "callId": message.get("toolCallId"),
                "status": "error" if message.get("isError") is True else "completed",
                "result": result_text(message.get("content")),
            }
            if payload.get("error") is not None:
                result["error"] = payload["error"]
            self.write(result)

    def finish(self, text: str, reason: dict[str, Any] | None) -> None:
        """Write the lossless terminal answer; it is exempt from the line cap."""
        self._sink(
            json.dumps(
                {
                    "type": "final",
                    "sessionId": self._session_id,
                    "reason": reason,
                    "text": text,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
