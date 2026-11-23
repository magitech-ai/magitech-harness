"""Phase 8 C4 conformance: recorded v4 fixtures replayed through the harness.

The fixtures under ``tests/fixtures/v4/`` are sanitized recorded v4 session
logs whose event shapes mirror the reference snapshot corpus (basic replay,
session titles, compaction checkpoint, approval audit). Values are neutral so
they remain stable across parity pins. Each fixture is validated by the real format
layer (``restore_released_v4_artifact`` + ``assert_released_v4_relationships``),
written to a JSONL root, and cold-loaded through ``SessionService`` — the same
seam the cross-tooling round-trip uses. Derived messages and projections are
the oracles: a fixture that stops loading or derives differently fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from magitech_harness import create_harness
from magitech_harness.session.events import KNOWN_SESSION_EVENT_TYPES
from magitech_harness.session.format import (
    SessionFormatArtifact,
    assert_released_v4_relationships,
    restore_released_v4_artifact,
)
from magitech_harness.session.persistence_jsonl import session_dir

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "v4"


def _read_fixture(name: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    raw = (_FIXTURES / name).read_text(encoding="utf-8").splitlines()
    header = json.loads(raw[0])
    rows = [json.loads(line) for line in raw[1:]]
    return header, rows, raw


def _validate_fixture(header: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Every committed fixture must pass the released v4 format layer."""
    artifact = restore_released_v4_artifact(
        SessionFormatArtifact(
            header={k: v for k, v in header.items() if k != "type"},
            inherited_event_count=0,
            events=rows,
        ),
        KNOWN_SESSION_EVENT_TYPES,
    )
    assert_released_v4_relationships(artifact, KNOWN_SESSION_EVENT_TYPES)


async def _load_fixture(root: Path, name: str) -> tuple[Any, Any, dict[str, Any]]:
    header, rows, raw = _read_fixture(name)
    _validate_fixture(header, rows)
    sid = header["id"]
    path = session_dir(root, header["cwd"], sid) / "session.v4.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(raw) + "\n", encoding="utf-8")
    ctx = create_harness(storage_dir=root)
    await ctx.flush()
    return ctx, ctx.sessions.load(sid), header


def _texts(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [
        (
            m.get("role", ""),
            "".join(
                b.get("text", "")
                for b in m.get("content", [])
                if isinstance(b, dict) and b.get("type") == "text"
            ),
        )
        for m in messages
    ]


@pytest.mark.asyncio
async def test_fixture_replay_basic_turn_and_tool_pair(tmp_path: Path) -> None:
    """A recorded basic turn replays to its exact model-visible message list."""
    _ctx, session, _ = await _load_fixture(tmp_path, "replay-basic.v4.jsonl")
    assert session is not None
    shape = _texts(session.derive_messages())
    assert shape == [
        ("system", "Be helpful."),
        ("user", "Inspect repo"),
        ("assistant", ""),  # tool-call block, no text
        ("tool", "ok"),
    ]
    # The tool result pairs with the assistant call (sourceEventSeqs cited).
    results = [e for e in session.events if e.type == "tool/result"]
    assert len(results) == 1
    assert results[0].source_event_seqs == [4]


@pytest.mark.asyncio
async def test_fixture_replay_title_projection(tmp_path: Path) -> None:
    """A recorded title fixture derives the pinned title (last write wins)."""
    ctx, session, _ = await _load_fixture(tmp_path, "titles.v4.jsonl")
    assert session is not None
    snapshot = ctx.sessionTitle.get(session)
    assert snapshot is not None
    assert snapshot["title"] == "Password reset steps"
    assert snapshot["source"]["kind"] == "provider"


@pytest.mark.asyncio
async def test_fixture_replay_compaction_checkpoint(tmp_path: Path) -> None:
    """A recorded checkpoint replaces the shadowed span in derived history."""
    _ctx, session, _ = await _load_fixture(tmp_path, "compaction.v4.jsonl")
    assert session is not None
    types = [e.type for e in session.events]
    assert types.count("compaction/start") == 1
    assert types.count("compaction/summary") == 1
    assert types.count("compaction/end") == 1
    checkpoint = next(
        e
        for e in session.events
        if e.type == "user/message"
        and (e.payload.get("source") or {}).get("kind") == "compact-checkpoint"
    )
    assert (checkpoint.surface_op or {}).get("op") == "replace"
    shape = _texts(session.derive_messages())
    # The old premise appears once, inside the checkpoint framing — not as a
    # standalone historical message. The tail and final reply survive.
    assert not any(text == "Establish a durable premise" for _, text in shape)
    assert any("<compacted-summary>" in text for _, text in shape)
    assert ("user", "Continue") in shape
    assert ("assistant", "COMPACTION RECOVERED") in shape


@pytest.mark.asyncio
async def test_fixture_replay_approval_audit_pairs(tmp_path: Path) -> None:
    """A recorded approval fixture replays both one-shot audit pairs."""
    _ctx, session, _ = await _load_fixture(tmp_path, "approval-audit.v4.jsonl")
    assert session is not None
    asked = [e for e in session.events if e.type == "approval/asked"]
    decided = [e for e in session.events if e.type == "approval/decided"]
    assert [e.payload["id"] for e in asked] == ["ap-1", "ap-2"]
    assert [e.payload["id"] for e in decided] == ["ap-1", "ap-2"]
    assert [e.payload["outcome"] for e in decided] == ["allowed-once", "rejected"]
    # Audit pairs are log-only: they never enter the derived message list.
    assert all(role != "approval" for role, _ in _texts(session.derive_messages()))
