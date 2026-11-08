"""Session storage and deterministic model history projection (derive_messages)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from magitech_harness.session.events import EventType, SessionEvent


def _normalize_adjacent_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merges adjacent messages with identical roles to satisfy strict model provider APIs."""
    if not messages:
        return []

    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not normalized:
            normalized.append(dict(msg))
            continue

        prev = normalized[-1]
        if (
            prev.get("role") == msg.get("role")
            and msg.get("role") in ("user", "system", "assistant")
            and "tool_calls" not in prev
            and "tool_calls" not in msg
        ):
            c1 = prev.get("content")
            c2 = msg.get("content")
            s1 = str(c1) if c1 is not None else ""
            s2 = str(c2) if c2 is not None else ""
            if s1 and s2:
                prev["content"] = f"{s1}\n\n{s2}"
            else:
                prev["content"] = s1 or s2
        else:
            normalized.append(dict(msg))

    return normalized


def derive_messages(events: list[SessionEvent]) -> list[dict[str, Any]]:
    """Deterministically reconstruct model-visible conversation history from session log.

    Adheres to: Model-visible ⟺ Logged.
    """
    latest_compaction: SessionEvent | None = None
    latest_compaction_idx = -1
    for i, ev in enumerate(events):
        if ev.type == EventType.COMPACTION:
            latest_compaction = ev
            latest_compaction_idx = i

    messages: list[dict[str, Any]] = []

    if latest_compaction is not None:
        summary = latest_compaction.payload.get("summary", "")
        messages.append(
            {
                "role": "user",
                "content": f"[Conversation summary up to checkpoint]:\n{summary}",
            }
        )
        compacted_count = latest_compaction.payload.get("compacted_count")
        if compacted_count is not None:
            slice_events = [
                ev for ev in events[compacted_count:] if ev.type != EventType.COMPACTION
            ]
        else:
            slice_events = events[latest_compaction_idx + 1 :]
    else:
        slice_events = events

    for ev in slice_events:
        if ev.type == EventType.USER_MESSAGE:
            messages.append(
                {
                    "role": "user",
                    "content": ev.payload.get("text", ""),
                }
            )
        elif ev.type == EventType.ASSISTANT_MESSAGE:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": ev.payload.get("text", "") or None,
            }
            tool_calls = ev.payload.get("tool_calls")
            if tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["call_id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                            if isinstance(tc["arguments"], dict)
                            else str(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(msg)
        elif ev.type == EventType.TOOL_RESULT:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ev.payload.get("call_id"),
                    "name": ev.payload.get("name"),
                    "content": json.dumps(ev.payload.get("content"))
                    if not isinstance(ev.payload.get("content"), str)
                    else ev.payload.get("content"),
                }
            )

    return _normalize_adjacent_messages(messages)


class Session:
    """A durable, append-only conversation session."""

    def __init__(self, session_id: str, storage_path: Path | None = None) -> None:
        self.session_id = session_id
        self.storage_path = storage_path
        self.events: list[SessionEvent] = []

    def append(self, event: SessionEvent) -> None:
        """Append event to in-memory log and persist to JSONL if configured."""
        if event.seq == 0:
            event.seq = len(self.events) + 1
        self.events.append(event)
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "a", encoding="utf-8") as f:
                try:
                    import fcntl

                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except (ImportError, OSError, AttributeError):
                    pass
                try:
                    f.write(event.model_dump_json() + "\n")
                    f.flush()
                finally:
                    try:
                        import fcntl

                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except (ImportError, OSError, AttributeError):
                        pass

    def derive_messages(self) -> list[dict[str, Any]]:
        """Project current model-visible messages from event log."""
        return derive_messages(self.events)

    @classmethod
    def load(cls, storage_path: Path, *, repair_on_open: bool = True) -> Session:
        """Load session from a JSONL file."""
        events: list[SessionEvent] = []
        session_id = storage_path.stem
        if storage_path.exists():
            with open(storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                        events.append(SessionEvent.model_validate(data))
                    except Exception:
                        if repair_on_open:
                            continue
                        raise
            if events:
                session_id = events[0].session_id
        sess = cls(session_id=session_id, storage_path=storage_path)
        sess.events = events
        return sess
