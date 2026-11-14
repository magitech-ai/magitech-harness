"""Historical tool definitions for request projection.

A request carries the baseline tool series plus the developer updates
recorded since the baseline, so a later reader can reconstruct exactly
which tool definitions a request was built under.
"""

from __future__ import annotations

import copy
from typing import Any

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.request_header import _canonical_json


def tool_name(schema: dict[str, Any]) -> str:
    """The name of one tool schema, read through its function envelope.

    Header tools are the envelope schemas returned by the tool service
    (``{"type": "function", "function": {"name", ...}}``); the name
    lives on the nested function object, not at the top level.
    """
    function = schema.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        if isinstance(name, str) and name:
            return name
    raise ValueError("tool history: header tool without a name")


def _tool_name(schema: Any) -> str:
    """Name one schema, refusing anything that is not a named envelope."""
    if not isinstance(schema, dict):
        raise TypeError("tool history: header tool without a name")
    return tool_name(schema)


class ToolHistoryProjection:
    """Stateful fold of headers and developer messages, in log order."""

    def __init__(self) -> None:
        # headerSeq -> the header's tools (detached copies).
        self._headers: dict[int, list[dict[str, Any]]] = {}
        # Definitions retained in the current declaration series.
        self._declared: dict[str, dict[str, Any]] = {}
        # Active definitions from the latest header.
        self._active: list[dict[str, Any]] = []
        # Names available in the series (baseline plus recorded updates).
        self._available: set[str] = set()
        self._baseline_seq: int | None = None
        self._history: dict[str, Any] = {"tools": [], "updates": []}

    def apply(self, event: SessionEvent) -> None:
        """Consume the next committed event in log order."""
        if event.type == "request/header":
            self._apply_header(event)
        elif event.type == "developer/message":
            self._apply_developer(event)

    def _apply_header(self, event: SessionEvent) -> None:
        header = event.payload.get("header")
        tools = header.get("tools") or [] if isinstance(header, dict) else []
        tools = copy.deepcopy(tools) if isinstance(tools, list) else []
        self._headers[event.seq] = tools
        redeclared = any(
            _tool_name(tool) in self._declared
            and _canonical_json(self._declared[_tool_name(tool)])
            != _canonical_json(tool)
            for tool in tools
        )
        if (
            self._baseline_seq is None
            or event.payload.get("reason") == "series"
            or event.payload.get("startsSeries") is True
            or redeclared
        ):
            self._baseline_seq = event.seq
            self._declared = {_tool_name(tool): tool for tool in tools}
            self._history = {"tools": list(tools), "updates": []}
            self._available = {_tool_name(tool) for tool in tools}
        self._active = list(tools)

    def _apply_developer(self, event: SessionEvent) -> None:
        message = event.payload.get("message")
        raw_content = message.get("content") if isinstance(message, dict) else None
        content: list[Any] = raw_content if isinstance(raw_content, list) else []
        header_seq = event.payload.get("headerSeq")
        definitions: list[dict[str, Any]] = (
            [] if header_seq is None else self._headers.get(header_seq) or []
        )
        additions: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool-addition":
                continue
            tool = next(
                (
                    candidate
                    for candidate in definitions
                    if _tool_name(candidate) == block.get("toolName")
                ),
                None,
            )
            if tool is None:
                raise ValueError(
                    f"tool history: missing definition for {block.get('toolName')}"
                )
            additions.append(tool)
        for tool in additions:
            self._declared[_tool_name(tool)] = tool
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool-addition":
                self._available.add(block["toolName"])
            elif block.get("type") == "tool-removal" and isinstance(
                block.get("toolName"), str
            ):
                self._available.discard(block["toolName"])
        message_id = message.get("id") if isinstance(message, dict) else None
        self._history = {
            "tools": self._history["tools"],
            "updates": [
                *self._history["updates"],
                {"messageId": message_id, "additions": additions},
            ],
        }

    def snapshot(self) -> dict[str, Any]:
        """The current request snapshot; a detached copy in both branches.

        When the active header's tool set and the available set disagree,
        the snapshot is the active header's tools with no updates: the
        request is built from what the latest header offers, and no
        historical addition is claimed.
        """
        if len(self._active) != len(self._available) or any(
            _tool_name(tool) not in self._available for tool in self._active
        ):
            return {"tools": copy.deepcopy(self._active), "updates": []}
        return copy.deepcopy(self._history)
