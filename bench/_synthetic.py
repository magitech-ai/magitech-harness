"""Deterministic synthetic v4 session logs for load/replay and fold workloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from magitech_harness.llm.message import (
    create_assistant_message,
    create_system_message,
    create_user_message,
)
from magitech_harness.session.events import KNOWN_SESSION_EVENT_TYPES
from magitech_harness.session.format import (
    SessionFormatArtifact,
    assert_released_v4_relationships,
    restore_released_v4_artifact,
)
from magitech_harness.session.persistence_jsonl import session_dir


def _row(t: str, seq: int, data: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"type": t, "seq": seq, "time": 1000 + seq, "data": data, **extra}


def build_session_rows(turns: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a valid v4 header + a full-turn log with ``turns`` user/assistant pairs."""
    header: dict[str, Any] = {
        "type": "session",
        "version": 4,
        "id": "bench-session",
        "createdAt": 1000,
        "cwd": "/work",
        "isSeeded": False,
        "delegationDepth": 0,
    }
    rows: list[dict[str, Any]] = []
    seq = 0

    def add(t: str, data: dict[str, Any], **extra: Any) -> None:
        nonlocal seq
        rows.append(_row(t, seq, data, **extra))
        seq += 1

    add("turn/start", {"turn": 1})
    add("step/start", {"turn": 1, "step": 1})
    add(
        "system/message",
        {
            "turn": 1,
            "step": 1,
            "message": create_system_message("Be a helpful benchmark."),
        },
        surfaceOp="append",
    )
    for i in range(1, turns + 1):
        add(
            "user/message",
            create_user_message(
                content=[
                    {"type": "text", "text": f"Question number {i} about the system"}
                ],
                source={"kind": "user"},
            ),
            surfaceOp="append",
        )
        add(
            "assistant/message",
            {
                "turn": 1,
                "step": 1,
                "message": create_assistant_message(
                    content=[
                        {"type": "text", "text": f"Answer number {i} with some detail"}
                    ],
                    provider="mock",
                    model="m1",
                ),
                "stream": [],
            },
            surfaceOp="append",
        )
    add("step/end", {"turn": 1, "step": 1})
    add("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    return header, rows


def write_session(root: Path, turns: int) -> Path:
    """Write a validated synthetic session to a JSONL root; return its path."""
    header, rows = build_session_rows(turns)
    artifact = restore_released_v4_artifact(
        SessionFormatArtifact(
            header={k: v for k, v in header.items() if k != "type"},
            inherited_event_count=0,
            events=rows,
        ),
        KNOWN_SESSION_EVENT_TYPES,
    )
    assert_released_v4_relationships(artifact, KNOWN_SESSION_EVENT_TYPES)
    path = session_dir(root, header["cwd"], header["id"]) / "session.v4.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header, *rows]
    path.write_text(
        "\n".join(json.dumps(r, separators=(",", ":")) for r in lines) + "\n",
        encoding="utf-8",
    )
    return path
