"""Durable projection state for loop-owned system-prompt and runtime-context messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from animas import Context

from magitech_harness.agent.system_prompt import (
    SystemPromptCommit,
    SystemPromptProjection,
)
from magitech_harness.llm.message import create_user_message
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.store import Session
from magitech_harness.session.surface import is_replacement_surface_event

_SOURCE = "runtime-context"
_CLEARED = (
    "Current runtime context: none. Earlier runtime-context snapshots no longer apply."
)
_UNSET = object()


def _is_owned(message: Mapping[str, Any]) -> bool:
    source = message.get("source")
    return isinstance(source, Mapping) and source.get("kind") == _SOURCE


def _text_of(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if not isinstance(content, Sequence) or len(content) != 1:
        return None
    block = content[0]
    if isinstance(block, Mapping) and block.get("type") == "text":
        text = block.get("text")
        return text if isinstance(text, str) else None
    return None


@dataclass(frozen=True)
class _RetainedSnapshot:
    seq: int
    text: str | None


class RuntimeContextProjection:
    """Tracks the last retained runtime-context snapshot without owning its commit."""

    def __init__(self, ctx: Context, session: Session) -> None:
        self._retained: _RetainedSnapshot | None | object = _UNSET
        surface = set(session.surface_nodes())
        for event in reversed(session.snapshot_events()):
            if event.type != "user/message" or not _is_owned(event.payload):
                continue
            if self._retained is _UNSET:
                self._retained = None
            if event.seq in surface:
                self._retained = _RetainedSnapshot(
                    seq=event.seq,
                    text=_text_of(event.payload),
                )
                break

        def _on_event(subject: Any, event: SessionEvent) -> None:
            if subject is not session:
                return
            if event.type == "user/message" and _is_owned(event.payload):
                self._retained = _RetainedSnapshot(
                    seq=event.seq,
                    text=_text_of(event.payload),
                )
            elif (
                isinstance(self._retained, _RetainedSnapshot)
                and is_replacement_surface_event(event)
                and event.source_event_seqs is not None
                and self._retained.seq in event.source_event_seqs
            ):
                self._retained = None

        ctx.on("session/event", _on_event)

    def project(
        self, current: str, sections: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        """Create an uncommitted snapshot only when the retained value differs."""
        if self._retained is _UNSET and current == "":
            return None
        snapshot = _CLEARED if current == "" else current
        if isinstance(self._retained, _RetainedSnapshot) and (
            self._retained.text == snapshot
        ):
            return None
        source: dict[str, Any] = (
            {"kind": _SOURCE}
            if len(sections) == 0
            else {"kind": _SOURCE, "form": "snapshot", "sections": list(sections)}
        )
        return create_user_message(
            content=[{"type": "text", "text": snapshot}],
            source=source,
        )


__all__ = [
    "RuntimeContextProjection",
    "SystemPromptCommit",
    "SystemPromptProjection",
]
