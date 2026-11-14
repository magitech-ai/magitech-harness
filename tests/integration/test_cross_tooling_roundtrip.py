"""Bidirectional v4 session log and projection checkpoint round-trip tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from anima import Context

from magitech_harness.session.events import KNOWN_SESSION_EVENT_TYPES
from magitech_harness.session.format import (
    SessionFormatArtifact,
    assert_released_v4_relationships,
    restore_released_v4_artifact,
)
from magitech_harness.session.persistence_jsonl import (
    JsonlSessionPersistenceService,
    session_dir,
)
from magitech_harness.session.projection_cache import (
    ProjectionCacheConfig,
    SessionProjectionCache,
)
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.projections import (
    SESSION_STATS_PROJECTION,
    TITLE_INPUT_PROJECTION,
    TITLE_PROJECTION,
    TURN_OUTLINE_PROJECTION,
    WORKING_DIRECTORY_PROJECTION,
)
from magitech_harness.session.service import SessionService


def _wire_header(session_id: str, created_at: int, cwd: str) -> dict[str, Any]:
    return {
        "type": "session",
        "version": 4,
        "id": session_id,
        "createdAt": created_at,
        "cwd": cwd,
        "isSeeded": False,
        "delegationDepth": 0,
    }


def _setup_service(
    root: Path,
) -> tuple[
    Context, SessionProjectionRegistry, JsonlSessionPersistenceService, SessionService
]:
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    for definition in (
        TITLE_PROJECTION,
        TITLE_INPUT_PROJECTION,
        SESSION_STATS_PROJECTION,
        TURN_OUTLINE_PROJECTION,
        WORKING_DIRECTORY_PROJECTION,
    ):
        registry.register(definition)
    persistence = JsonlSessionPersistenceService(ctx, {"root": root})
    service = SessionService(ctx, {"base_storage_dir": root})
    return ctx, registry, persistence, service


@pytest.mark.asyncio
async def test_bidirectional_v4_session_roundtrip(tmp_path: Path) -> None:
    """Harness-written v4 JSONL parses cleanly and external v4 logs load back identically."""
    root_a = tmp_path / "harness-a"
    root_b = tmp_path / "harness-b"
    _, registry_a, _, service_a = _setup_service(root_a)

    session = service_a.create("xt-1")
    service_a.log_event(
        session,
        "request/header",
        {
            "header": {"config": {"provider": "mock", "model": "m-1"}},
            "reason": "initial",
        },
    )
    service_a.log_event(session, "turn/start", {"turn": 1})
    u1 = service_a.log_event(
        session,
        "user/message",
        {
            "content": [{"type": "text", "text": "Inspect repo"}],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )
    service_a.log_event(session, "step/start", {"turn": 1, "step": 1})
    service_a.log_event(
        session,
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "a-1",
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "thinking"},
                    {
                        "type": "tool-call",
                        "toolCallId": "tc-1",
                        "name": "read",
                        "arguments": "{}",
                    },
                ],
                "source": {"kind": "model", "provider": "mock", "model": "m-1"},
            },
            "stream": [],
        },
        surface_op="append",
    )
    service_a.log_event(
        session,
        "tool/result",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "tr-1",
                "role": "tool",
                "toolCallId": "tc-1",
                "isError": False,
                "source": {"kind": "tool", "callId": "tc-1"},
                "content": [{"type": "text", "text": "ok"}],
            },
        },
        surface_op="append",
    )
    service_a.log_event(session, "step/end", {"turn": 1, "step": 1})
    service_a.log_event(
        session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}
    )
    service_a.log_event(
        session,
        "session/title",
        {"title": "Repo inspect", "messageSeqs": [u1.seq], "source": {"kind": "user"}},
    )
    await service_a.flush(session)
    expected_messages = session.derive_messages()
    expected_snapshot = registry_a.snapshot(session)
    log_path = service_a.location("xt-1")
    assert log_path is not None
    assert service_a.close("xt-1") is True
    cold_a = service_a.read("xt-1")
    assert cold_a is not None
    expected_wire_snapshot = registry_a.snapshot(cold_a)

    # Inspect raw wire JSONL written on disk: valid v4 envelope and relationships.
    raw_lines = log_path.read_text(encoding="utf-8").splitlines()
    raw_header = json.loads(raw_lines[0])
    assert raw_header["type"] == "session"
    assert raw_header["version"] == 4
    wire_events = [json.loads(line) for line in raw_lines[1:]]
    assert [ev["seq"] for ev in wire_events] == list(range(len(wire_events)))

    # Write a byte-equivalent external v4 log into root_b and open it cleanly.
    ext_log = session_dir(root_b, raw_header.get("cwd"), "xt-1") / "session.v4.jsonl"
    ext_log.parent.mkdir(parents=True, exist_ok=True)
    header_no_type = {k: v for k, v in raw_header.items() if k != "type"}
    artifact = restore_released_v4_artifact(
        SessionFormatArtifact(
            header=header_no_type,
            inherited_event_count=0,
            events=wire_events,
        ),
        KNOWN_SESSION_EVENT_TYPES,
    )
    assert_released_v4_relationships(artifact, KNOWN_SESSION_EVENT_TYPES)
    ext_log.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    _, registry_b, _, service_b = _setup_service(root_b)
    reloaded = service_b.load("xt-1")
    assert reloaded is not None
    assert reloaded.derive_messages() == expected_messages
    assert registry_b.snapshot(reloaded) == expected_wire_snapshot
    assert expected_snapshot["values"]["title"] == "Repo inspect"
    assert (
        expected_snapshot["values"]["turnOutline"]
        == expected_wire_snapshot["values"]["turnOutline"]
    )
    assert service_b.close("xt-1") is True


@pytest.mark.asyncio
async def test_bidirectional_v4_catalog_team_and_projection_roundtrip(
    tmp_path: Path,
) -> None:
    """v4 team/mode, subagent/catalog v3, and projection checkpoints round-trip cleanly."""
    root = tmp_path / "ext"
    workdir = str((tmp_path / "proj").resolve())
    created_at = 1700000100000
    log_path = session_dir(root, workdir, "xt-team") / "session.v4.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    wire_rows = [
        _wire_header("xt-team", created_at, workdir),
        {
            "type": "team/mode",
            "seq": 0,
            "time": created_at + 1,
            "data": {"enabled": True},
        },
        {
            "type": "subagent/catalog",
            "seq": 1,
            "time": created_at + 2,
            "data": {
                "version": 3,
                "childId": "worker-1",
                "childCreatedAt": created_at + 2,
                "mode": "continuable",
                "label": "Reviewer",
                "name": "reviewer",
            },
        },
        {
            "type": "user/message",
            "seq": 2,
            "time": created_at + 3,
            "surfaceOp": "append",
            "data": {
                "content": [{"type": "text", "text": "Coordinate review"}],
                "source": {"kind": "user"},
            },
        },
    ]
    ext_artifact = restore_released_v4_artifact(
        SessionFormatArtifact(
            header={k: v for k, v in wire_rows[0].items() if k != "type"},
            inherited_event_count=0,
            events=wire_rows[1:],
        ),
        KNOWN_SESSION_EVENT_TYPES,
    )
    assert_released_v4_relationships(ext_artifact, KNOWN_SESSION_EVENT_TYPES)
    log_path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in wire_rows) + "\n",
        encoding="utf-8",
    )

    _, registry, _, service = _setup_service(root)
    cache = SessionProjectionCache(
        registry,
        root,
        ProjectionCacheConfig(write_every_events=200, write_interval_ms=5000),
    )
    loaded = service.load("xt-team")
    assert loaded is not None
    assert [ev.type for ev in loaded.events] == [
        "team/mode",
        "subagent/catalog",
        "user/message",
    ]
    assert loaded.derive_messages() == [
        {
            "content": [{"type": "text", "text": "Coordinate review"}],
            "source": {"kind": "user"},
        }
    ]
    cold = cache.cold_snapshot(
        loaded.header, loaded.inherited_event_count, list(loaded.events)
    )
    assert cold["as_of_seq"] == 2
    assert cold["values"]["workingDirectory"] == workdir
    assert service.close("xt-team") is True
    cache.dispose()

    # Reload persisted projection cache from disk with a fresh cache instance.
    reloaded_cache = SessionProjectionCache(
        registry,
        root,
        ProjectionCacheConfig(write_every_events=200, write_interval_ms=5000),
    )
    cached = reloaded_cache.cached_snapshot(loaded.header)
    assert cached is not None
    assert cached["as_of_seq"] == 2
    assert cached["values"]["workingDirectory"] == workdir
    reloaded_cache.dispose()
