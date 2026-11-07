"""
Session storage and deterministic model history projection (derive_messages).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from magitech_harness.session.events import EventType, SessionEvent


def derive_messages(events: list[SessionEvent]) -> list[dict[str, Any]]:
    """Deterministically reconstruct model-visible conversation history from session log.
    
    Adheres to: Model-visible ⟺ Logged.
    """
    messages: list[dict[str, Any]] = []

    for ev in events:
        if ev.type == EventType.USER_MESSAGE:
            messages.append({
                "role": "user",
                "content": ev.payload.get("text", ""),
            })
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
            messages.append({
                "role": "tool",
                "tool_call_id": ev.payload.get("call_id"),
                "name": ev.payload.get("name"),
                "content": json.dumps(ev.payload.get("content"))
                if not isinstance(ev.payload.get("content"), str)
                else ev.payload.get("content"),
            })

    return messages


class Session:
    """A durable, append-only conversation session."""

    def __init__(self, session_id: str, storage_path: Path | None = None) -> None:
        self.session_id = session_id
        self.storage_path = storage_path
        self.events: list[SessionEvent] = []

    def append(self, event: SessionEvent) -> None:
        """Append event to in-memory log and persist to JSONL if configured."""
        self.events.append(event)
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(event.model_dump_json() + "\n")

    def derive_messages(self) -> list[dict[str, Any]]:
        """Project current model-visible messages from event log."""
        return derive_messages(self.events)

    @classmethod
    def load(cls, storage_path: Path) -> Session:
        """Load session from a JSONL file."""
        events: list[SessionEvent] = []
        session_id = storage_path.stem
        if storage_path.exists():
            with open(storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        events.append(SessionEvent.model_validate(data))
            if events:
                session_id = events[0].session_id
        sess = cls(session_id=session_id, storage_path=storage_path)
        sess.events = events
        return sess
