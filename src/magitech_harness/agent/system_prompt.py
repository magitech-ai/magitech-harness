"""System-prompt projection: how a rendered prompt reaches the surface.

Pure loop policy. It takes the rendered prompt and the route/series
facts and returns ordered commits; it never appends. The loop owns the
commit. The first prompt, even empty, reserves surface node 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from magitech_harness.session.messages import create_system_message


@dataclass(frozen=True)
class SystemPromptCommit:
    """One uncommitted system-prompt surface operation."""

    message: dict[str, Any]
    intent: dict[str, Any]


@dataclass(frozen=True)
class _SystemNode:
    seq: int
    # Effective text: "" for empty content, None when the node's content
    # is not a single text block (a multi-block message has no comparable text).
    text: str | None


class SystemPromptProjection:
    """Decides how a rendered system prompt reaches the surface.

    Constructed with a reader over the session returning the surviving
    ``system/message`` nodes in surface order as ``{"seq", "text"}``
    records.
    """

    def __init__(self, system_nodes: Callable[[], list[dict[str, Any]]]) -> None:
        self._system_nodes = system_nodes

    def project(
        self, rendered: str, *, in_history: bool, starts_series: bool
    ) -> list[SystemPromptCommit]:
        """Return the ordered commits for one rendered prompt."""
        nodes = [
            _SystemNode(seq=node["seq"], text=node["text"])
            for node in self._system_nodes()
        ]
        head = nodes[0] if nodes else None
        if head is None:
            return [
                SystemPromptCommit(
                    message=create_system_message(rendered),
                    intent={"surface_op": "append"},
                )
            ]
        latest = next((node for node in reversed(nodes) if node.text != ""), head)
        if (
            not in_history
            or rendered == ""
            or (starts_series and latest.text != rendered)
        ):
            commits = [
                self._replace(node.seq, "") for node in nodes[1:] if node.text != ""
            ]
            if head.text != rendered:
                commits.append(self._replace(head.seq, rendered))
            return commits
        if latest.text == rendered:
            return []
        return [
            SystemPromptCommit(
                message=create_system_message(rendered),
                intent={"surface_op": "append"},
            )
        ]

    @staticmethod
    def _replace(seq: int, text: str) -> SystemPromptCommit:
        return SystemPromptCommit(
            message=create_system_message(text),
            intent={
                "surface_op": {"op": "replace", "startSeq": seq, "endSeq": seq},
                "source_event_seqs": [seq],
            },
        )
