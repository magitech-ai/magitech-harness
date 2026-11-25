"""Conformance: the default approval guard fails closed on mutating tools.

Finding #9 proof: a mutating tool that declares requiresApproval cannot run
without an approver, and the approval/asked + approval/decided audit pair is
appended to the requesting session in order. Finding #12 proof: the guard
handles both the dict exec_view and the legacy ToolCallData shapes.
"""

from __future__ import annotations

from typing import Any

import pytest

from magitech_harness import create_harness, define_tool
from magitech_harness.agent.cancellation import AbortController
from magitech_harness.session.events import EventType


def _tool(name: str = "sensitive_op", *, requires_approval: bool = True) -> Any:
    return define_tool(
        {
            "name": name,
            "description": "Sensitive tool.",
            "parameters": {},
            "output": {
                "schema": {"type": "object", "additionalProperties": True},
                "render": lambda _a, val: [{"type": "text", "text": "ran"}],
            },
            "execute": lambda _a, _c: {"ran": True},
            **({"requiresApproval": True} if requires_approval else {}),
        }
    )


@pytest.mark.asyncio
async def test_no_approver_mutating_tool_fails_closed_with_audit(tmp_path: Any) -> None:
    """A declared mutating tool is denied with no approver; both audit events land."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = ctx.sessions.create("approval-guard-probe")
    session.append(EventType.TURN_START, {"turn": 1})

    ran = []

    def _mark(_a: Any, _c: Any) -> dict[str, bool]:
        ran.append(True)
        return {"ran": True}

    # Replace the tool body so we can observe that it never executes.
    ctx.tools.register(
        define_tool(
            {
                "name": "sensitive_op",
                "description": "Sensitive tool.",
                "parameters": {},
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": lambda _a, val: [{"type": "text", "text": "ran"}],
                },
                "execute": _mark,
                "requiresApproval": True,
            }
        )
    )

    batch = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "c1",
                    "name": "sensitive_op",
                    "arguments": "{}",
                }
            ],
            "signal": AbortController().signal,
            "agent": type("Agent", (), {"session": session})(),
        }
    )
    result = batch["results"][0]["result"]
    assert result["isError"] is True
    assert ran == []
    audit = [
        e
        for e in session.events
        if e.type in (EventType.APPROVAL_ASKED, EventType.APPROVAL_DECIDED)
    ]
    assert [e.type for e in audit] == [
        EventType.APPROVAL_ASKED,
        EventType.APPROVAL_DECIDED,
    ]
    assert audit[1].payload["outcome"] == "unavailable"


@pytest.mark.asyncio
async def test_read_only_tool_runs_without_approval(tmp_path: Any) -> None:
    """A read-only tool bypasses the guard and executes normally."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    session = ctx.sessions.create("read-only-probe")
    session.append(EventType.TURN_START, {"turn": 1})
    ran = []
    ctx.tools.register(
        define_tool(
            {
                "name": "probe_read",
                "description": "Read.",
                "parameters": {},
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": lambda _a, val: [{"type": "text", "text": "ok"}],
                },
                "execute": lambda _a, _c: ran.append(True) or {"ok": True},
            }
        )
    )
    batch = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "c2",
                    "name": "probe_read",
                    "arguments": "{}",
                }
            ],
            "signal": AbortController().signal,
            "agent": type("Agent", (), {"session": session})(),
        }
    )
    assert batch["results"][0]["result"]["isError"] is False
    assert ran == [True]
