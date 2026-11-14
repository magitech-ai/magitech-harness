"""Unit tests for the projection registry."""

from typing import Any

import pytest
from anima import Context

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    CheckpointRow,
    ProjectionDefinition,
    SessionProjectionRegistry,
    UnitCell,
    WireView,
)
from magitech_harness.session.projections import (
    SESSION_STATS_PROJECTION,
    TITLE_INPUT_PROJECTION,
    TITLE_PROJECTION,
    TURN_OUTLINE_PROJECTION,
    WORKING_DIRECTORY_PROJECTION,
    validate_title_value,
)
from magitech_harness.session.store import Session

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "s1",
    "createdAt": 1700000000000,
    "cwd": "/home/malcom/work",
    "isSeeded": False,
    "delegationDepth": 0,
}

ALL_DEFINITIONS = (
    TITLE_PROJECTION,
    TITLE_INPUT_PROJECTION,
    SESSION_STATS_PROJECTION,
    TURN_OUTLINE_PROJECTION,
    WORKING_DIRECTORY_PROJECTION,
)


def _event(
    event_type: str,
    payload: dict[str, Any],
    seq: int,
    ms: int,
    *,
    surface_op: str | None = None,
) -> SessionEvent:
    return SessionEvent(
        id=f"s1:{seq}",
        session_id="s1",
        type=event_type,
        timestamp=ms / 1000,
        payload=payload,
        seq=seq,
        surface_op=surface_op,
    )


def _user_message(text: str, seq: int, ms: int) -> SessionEvent:
    return _event(
        "user/message",
        {
            "content": [{"type": "text", "text": text}],
            "source": {"kind": "user"},
        },
        seq,
        ms,
        surface_op="append",
    )


def _assistant_message(
    text: str,
    seq: int,
    ms: int,
    *,
    stream: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> SessionEvent:
    payload: dict[str, Any] = {
        "turn": 1,
        "step": 1,
        "message": {
            "id": f"a{seq}",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }
    if stream is not None:
        payload["stream"] = stream
    if usage is not None:
        payload["usage"] = usage
    return _event("assistant/message", payload, seq, ms, surface_op="append")


def _golden_events() -> list[SessionEvent]:
    """The spec §5.1 Scenario A log (10 events, exact timestamps)."""
    return [
        _event("turn/start", {"turn": 1}, 0, 1000),
        _user_message("Hello world", 1, 1100),
        _event("step/start", {"turn": 1, "step": 1}, 2, 1200),
        _assistant_message(
            "Hi there",
            3,
            1500,
            stream=[
                {
                    "type": "chunk",
                    "time": 1300,
                    "chunk": {"type": "text-delta", "index": 0, "text": "Hi"},
                }
            ],
            usage={"outputTokens": 5},
        ),
        _event(
            "tool/call",
            {"turn": 1, "step": 1, "callId": "c1", "name": "add", "arguments": {}},
            4,
            1600,
        ),
        _event(
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "m1",
                    "role": "tool",
                    "source": {"kind": "tool", "callId": "c1"},
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "c1",
                            "content": [{"type": "text", "text": "3"}],
                        }
                    ],
                },
            },
            5,
            1800,
            surface_op="append",
        ),
        _event("step/end", {"turn": 1, "step": 1}, 6, 1900),
        _event("turn/end", {"turn": 1, "reason": {"kind": "completed"}}, 7, 2000),
        _event(
            "session/title",
            {
                "title": "Hello world",
                "messageSeqs": [1],
                "source": {"kind": "fallback"},
            },
            8,
            2100,
        ),
        _event("working-directory/change", {"cwd": "/tmp/proj"}, 9, 2200),
    ]


ALL_KEYS = ("title", "titleInput", "sessionStats", "turnOutline", "workingDirectory")


def _registry() -> tuple[Context, SessionProjectionRegistry]:
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    for definition in ALL_DEFINITIONS:
        registry.register(definition)
    return ctx, registry


def _require_int(value: Any) -> int:
    """A strict int validator for the counting test unit."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("not an int")
    return value


def _counting_definition() -> tuple[ProjectionDefinition[int], list[int]]:
    """A unit that counts its apply calls (proves lazy/drive paths)."""
    calls: list[int] = []

    def _apply(state: int, event: SessionEvent) -> int:
        calls.append(event.seq)
        return state + 1

    definition: ProjectionDefinition[int] = ProjectionDefinition(
        key="counter",
        state_schema=_require_int,
        init=lambda header, inherited: 0,
        apply=_apply,
        wire=WireView(view_schema=_require_int, view=lambda state: state),
        state_version=1,
    )
    return definition, calls


def _session_with_golden(ctx: Context) -> Session:
    events = _golden_events()
    return Session.create("s1", seed=list(events), header=dict(HEADER))


def _drive_all(ctx: Context, session: Session) -> None:
    for event in session.events:
        ctx.emit("session/event", session, event)


# ---------------------------------------------------------------------------
# Registration, refcounting, conflicts.
# ---------------------------------------------------------------------------


def test_register_validates_state_version() -> None:
    """A state version must be a non-negative int, else registration throws."""
    _, registry = _registry()
    for bad in (-1, 1.5, True, "1", None):
        with pytest.raises(ValueError):
            registry.register(
                ProjectionDefinition(
                    key="bad",
                    state_schema=lambda value: value,
                    init=lambda header, inherited: None,
                    apply=lambda state, event: state,
                    state_version=bad,  # type: ignore[arg-type]
                )
            )
    assert "bad" not in registry.summary_keys()


def test_register_refcounts_and_final_dispose_removes_key() -> None:
    """Same key + same version refcounts; the last release removes it."""
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    original = registry.register(TITLE_PROJECTION)
    first = registry.register(TITLE_PROJECTION)
    second = registry.register(TITLE_PROJECTION)
    assert registry.summary_keys() == frozenset({"title"})
    first.dispose()
    assert "title" in registry.summary_keys()
    second.dispose()
    assert "title" in registry.summary_keys()
    original.dispose()
    assert "title" not in registry.summary_keys()
    third = registry.register(TITLE_PROJECTION)
    assert "title" in registry.summary_keys()
    third.dispose()
    assert "title" not in registry.summary_keys()


def test_register_conflicts_raise() -> None:
    """Same key at another version or another summary throws."""
    _, registry = _registry()
    with pytest.raises(ValueError):
        registry.register(
            ProjectionDefinition(
                key="title",
                state_schema=validate_title_value,
                init=lambda header, inherited: None,
                apply=lambda state, event: state,
                wire=WireView(view_schema=validate_title_value, view=lambda s: s),
                state_version=2,
            )
        )
    with pytest.raises(ValueError):
        registry.register(
            ProjectionDefinition(
                key="sessionStats",
                state_schema=lambda value: value,
                init=lambda header, inherited: {},
                apply=lambda state, event: state,
                wire=WireView(
                    view_schema=lambda value: value,
                    view=lambda state: state,
                    summary=True,
                ),
                state_version=1,
            )
        )
    # Host-only vs wired is a summary conflict too (None vs True).
    with pytest.raises(ValueError):
        registry.register(
            ProjectionDefinition(
                key="titleInput",
                state_schema=lambda value: value,
                init=lambda header, inherited: {},
                apply=lambda state, event: state,
                wire=WireView(view_schema=lambda value: value, view=lambda s: s),
                state_version=3,
            )
        )


def test_handle_state_of_survives_disposal() -> None:
    """A retained handle keeps folding new events after unregistration."""
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    handle = registry.register(TITLE_PROJECTION)
    session = _session_with_golden(ctx)
    _drive_all(ctx, session)
    assert handle.state_of(session) == "Hello world"
    handle.dispose()
    assert "title" not in registry.summary_keys()
    assert registry.cached_snapshot(session) is None
    assert handle.state_of(session) == "Hello world"
    reader = handle.reader_for(session)
    assert reader() == "Hello world"
    handle.dispose()  # second dispose is a no-op, not a second release.
    assert handle.state_of(session) == "Hello world"
    # New events still fold on demand through the retained handle.
    session.append(
        "session/title",
        {"title": "Renamed", "messageSeqs": [], "source": {"kind": "user"}},
    )
    assert handle.state_of(session) == "Renamed"


def test_summary_keys_is_stable_until_the_table_changes() -> None:
    """summary_keys returns the same instance until register/dispose."""
    _, registry = _registry()
    first = registry.summary_keys()
    assert registry.summary_keys() is first
    # A refcount increment changes no key: the instance is stable.
    handle = registry.register(TITLE_PROJECTION)
    assert registry.summary_keys() is first
    handle.dispose()
    assert registry.summary_keys() is first
    # A genuinely new key rebuilds the set.
    definition, _ = _counting_definition()
    counter = registry.register(definition)
    rebuilt = registry.summary_keys()
    assert rebuilt is not first
    assert rebuilt == frozenset({"title", "workingDirectory", "counter"})
    assert registry.summary_keys() is rebuilt
    counter.dispose()
    assert registry.summary_keys() == frozenset({"title", "workingDirectory"})


# ---------------------------------------------------------------------------
# snapshot / cached_snapshot / checkpoint.
# ---------------------------------------------------------------------------


def test_snapshot_key_selection() -> None:
    """Keys may be a list (copied) or a set (used as given); None is all."""
    ctx, registry = _registry()
    session = _session_with_golden(ctx)
    _drive_all(ctx, session)
    full = registry.snapshot(session)
    assert set(full["values"]) == {
        "title",
        "sessionStats",
        "turnOutline",
        "workingDirectory",
    }
    assert registry.snapshot(session, ["title"])["values"] == {"title": "Hello world"}
    keys = registry.summary_keys()
    assert registry.snapshot(session, keys)["values"] == {
        "title": "Hello world",
        "workingDirectory": "/tmp/proj",
    }
    assert registry.snapshot(session, {"nope"})["values"] == {}


def test_cached_snapshot_is_none_without_cells() -> None:
    """No materialized cell means no cached snapshot — and no folding."""
    _, registry = _registry()
    session = Session.create("s1", seed=list(_golden_events()), header=dict(HEADER))
    assert registry.cached_snapshot(session) is None
    # Reading the live snapshot materializes; the cached one then agrees.
    assert registry.snapshot(session)["as_of_seq"] == 10
    assert registry.cached_snapshot(session) == registry.snapshot(session)


def test_cached_snapshot_lowest_watermark() -> None:
    """cached_snapshot's as_of_seq is the lowest served watermark."""
    ctx, registry = _registry()
    session = _session_with_golden(ctx)
    _drive_all(ctx, session)
    # Append three events without publication: every cell goes stale at 10.
    session.append("approval/policy", {"policy": "x"})
    session.append("approval/policy", {"policy": "y"})
    session.append(
        "session/title",
        {"title": "New", "messageSeqs": [], "source": {"kind": "user"}},
    )
    # Advance only the title cell; the rest stay at 10.
    title_handle = registry.register(TITLE_PROJECTION)
    assert title_handle.state_of(session) == "New"
    cached = registry.cached_snapshot(session)
    assert cached is not None
    assert cached["as_of_seq"] == 10
    assert cached["values"]["title"] == "New"
    # The live snapshot advances every cell to the shared cursor.
    assert registry.snapshot(session)["as_of_seq"] == 13
    assert registry.cached_snapshot(session)["as_of_seq"] == 13
    # A key selection skips unselected cells without folding.
    assert registry.cached_snapshot(session, ["title"]) == {
        "as_of_seq": 13,
        "values": {"title": "New"},
    }
    assert registry.cached_snapshot(session, ["nope"]) is None


def test_checkpoint_rows_are_detached() -> None:
    """Mutating a checkpoint row must not corrupt the live cell."""
    ctx, registry = _registry()
    session = _session_with_golden(ctx)
    _drive_all(ctx, session)
    rows = registry.checkpoint(session)
    assert rows["title"] == {"ver": 1, "seq": 10, "val": "Hello world"}
    assert rows["titleInput"]["ver"] == 3
    rows["sessionStats"]["val"]["turns"] = 99
    rows["turnOutline"]["val"]["turns"].append(
        {"turn": 9, "seq": 99, "prompt": "", "response": ""}
    )
    assert registry.snapshot(session)["values"]["sessionStats"]["turns"] == 1
    assert len(registry.snapshot(session)["values"]["turnOutline"]) == 1


def test_snapshot_invalid_view_raises() -> None:
    """A view that fails its schema raises instead of serving a bad value."""
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    registry.register(
        ProjectionDefinition(
            key="broken",
            state_schema=lambda value: value,
            init=lambda header, inherited: "ok",
            apply=lambda state, event: state,
            wire=WireView(
                view_schema=lambda value: (
                    value
                    if isinstance(value, str) and value
                    else (_ for _ in ()).throw(ValueError("bad view"))
                ),
                view=lambda state: "",
            ),
            state_version=0,
        )
    )
    session = Session.create("s1", seed=[], header=dict(HEADER))
    with pytest.raises(ValueError):
        registry.snapshot(session)


# ---------------------------------------------------------------------------
# restore_floor / restore / view_checkpoint.
# ---------------------------------------------------------------------------


def test_restore_floor_no_registrations_is_none() -> None:
    """With no units registered there is no tail read to do."""
    registry = SessionProjectionRegistry(Context())
    assert registry.restore_floor({}) is None


def test_restore_floor_one_below_lowest_usable() -> None:
    """A mismatched or absent row pulls the floor to 0; else one below."""
    _, registry = _registry()
    assert registry.restore_floor({}) == 0
    rows: dict[str, CheckpointRow] = {
        key: CheckpointRow(ver=1, seq=9, val=None) for key in ALL_KEYS
    }
    rows["titleInput"] = CheckpointRow(ver=3, seq=9, val=None)
    rows["turnOutline"] = CheckpointRow(ver=2, seq=9, val=None)
    assert registry.restore_floor(rows) == 9
    rows["title"] = CheckpointRow(ver=99, seq=9, val=None)
    assert registry.restore_floor(rows) == 0
    del rows["title"]
    assert registry.restore_floor(rows) == 0
    rows["title"] = CheckpointRow(ver=1, seq=-1, val=None)
    rows["titleInput"] = CheckpointRow(ver=3, seq=4, val=None)
    assert registry.restore_floor(rows) == 0
    # Malformed rows are unusable, never fatal.
    rows["title"] = {"ver": True, "seq": 9, "val": None}  # type: ignore[dict-item]
    assert registry.restore_floor(rows) == 0
    rows["title"] = {"ver": 1, "seq": "9", "val": None}  # type: ignore[dict-item]
    assert registry.restore_floor(rows) == 0
    rows["title"] = "not-a-row"  # type: ignore[dict-item]
    assert registry.restore_floor(rows) == 0


def test_restore_discards_version_mismatch_at_base_zero() -> None:
    """A mismatched row at base 0 is discarded and the log refolds."""
    _, registry = _registry()
    events = _golden_events()
    rows: dict[str, CheckpointRow] = {
        "title": CheckpointRow(ver=99, seq=8, val="Stale title"),
    }
    restored = registry.restore(rows, events, 0, HEADER, 0)
    assert restored.snapshot["values"]["title"] == "Hello world"
    assert restored.checkpoint["title"] == {"ver": 1, "seq": 9, "val": "Hello world"}


def test_restore_unusable_row_above_base_raises() -> None:
    """An unusable row with base_seq > 0 demands a re-read from seq 0."""
    _, registry = _registry()
    events = _golden_events()[5:]
    rows: dict[str, CheckpointRow] = {
        "title": CheckpointRow(ver=99, seq=8, val="Stale title"),
    }
    with pytest.raises(ValueError, match="re-read from seq 0"):
        registry.restore(rows, events, 5, HEADER, 0)
    with pytest.raises(ValueError, match="re-read from seq 0"):
        registry.restore({}, events, 5, HEADER, 0)
    # A version-correct row that predates the tail is unusable the same way.
    with pytest.raises(ValueError, match="re-read from seq 0"):
        registry.restore(
            {"title": CheckpointRow(ver=1, seq=2, val="Old title")},
            events,
            5,
            HEADER,
            0,
        )


def test_restore_malformed_seq_rows_are_discarded_at_base_zero() -> None:
    """Rows with a non-integer seq are unusable and refold at base 0."""
    _, registry = _registry()
    events = _golden_events()
    for bad_seq in (True, "8", 1.5):
        rows: dict[str, CheckpointRow] = {
            "title": CheckpointRow(ver=1, seq=bad_seq, val="Stale"),  # type: ignore[dict-item]
        }
        restored = registry.restore(rows, events, 0, HEADER, 0)
        assert restored.snapshot["values"]["title"] == "Hello world"


def test_restore_row_beyond_log_end_is_discarded_at_base_zero() -> None:
    """A row claiming events past the supplied end refolds from the log."""
    _, registry = _registry()
    events = _golden_events()[:6]
    rows: dict[str, CheckpointRow] = {
        "title": CheckpointRow(ver=1, seq=9, val="Hello world"),
    }
    restored = registry.restore(rows, events, 0, HEADER, 0)
    assert restored.snapshot["values"]["title"] is None
    assert restored.snapshot["as_of_seq"] == 5


def test_restore_missing_seq_raises() -> None:
    """A gap in the supplied tail raises instead of folding across it."""
    _, registry = _registry()
    events = _golden_events()
    gapped = [events[0], events[1], events[3]]
    with pytest.raises(ValueError, match="missing seq 2"):
        registry.restore({}, gapped, 0, HEADER, 0)


def test_restore_invalid_persisted_state_raises() -> None:
    """Persisted state failing its schema raises on restore (fail loud)."""
    _, registry = _registry()
    events = _golden_events()
    # Empty title: valid fold output, invalid persisted state (OQ-2).
    rows: dict[str, CheckpointRow] = {
        "title": CheckpointRow(ver=1, seq=8, val=""),
    }
    with pytest.raises(ValueError):
        registry.restore(rows, events, 0, HEADER, 0)
    rows = {
        "sessionStats": CheckpointRow(
            ver=1,
            seq=9,
            val={
                "turns": 1,
                "steps": 1,
                "llm_ms": -1.0,
                "tool_ms": 0.0,
                "ttft_ms": 0.0,
                "ttft_steps": 0,
                "decode_ms": 0.0,
                "decode_tokens": 0,
                "last_turn": 1,
                "open_step": None,
                "pending_calls": {},
            },
        )
    }
    with pytest.raises(ValueError):
        registry.restore(rows, events, 0, HEADER, 0)


def test_restore_empty_tail() -> None:
    """An empty tail restores the init states at base_seq - 1."""
    _, registry = _registry()
    restored = registry.restore({}, [], 0, HEADER, 0)
    assert restored.snapshot == {
        "as_of_seq": -1,
        "values": {
            "title": None,
            "sessionStats": {
                "turns": 0,
                "steps": 0,
                "llm_ms": 0.0,
                "tool_ms": 0.0,
                "ttft_ms": 0.0,
                "ttft_steps": 0,
                "decode_ms": 0.0,
                "decode_tokens": 0,
            },
            "turnOutline": [],
            "workingDirectory": "/home/malcom/work",
        },
    }
    assert restored.checkpoint["title"] == {"ver": 1, "seq": -1, "val": None}


def test_view_checkpoint_skips_bad_rows() -> None:
    """Mismatched, malformed, or invalid rows leave their key absent."""
    _, registry = _registry()
    rows: dict[str, CheckpointRow] = {
        "title": CheckpointRow(ver=1, seq=8, val="Hello world"),
        "sessionStats": CheckpointRow(ver=99, seq=9, val={}),
        "turnOutline": CheckpointRow(ver=2, seq=9, val={"turns": [], "draft": ""}),
        "workingDirectory": CheckpointRow(ver=1, seq=9, val="relative"),
    }
    assert registry.view_checkpoint(rows) == {
        "title": "Hello world",
        "turnOutline": [],
    }
    # An empty-title row fails the title schema and is skipped, not raised.
    assert (
        registry.view_checkpoint({"title": CheckpointRow(ver=1, seq=8, val="")}) == {}
    )
    # Malformed rows are skipped, never fatal.
    assert registry.view_checkpoint({"title": "not-a-row"}) == {}  # type: ignore[dict-item]
    assert (
        registry.view_checkpoint({"title": {"ver": True, "seq": 8, "val": "x"}}) == {}
    )  # type: ignore[dict-item]
    assert registry.view_checkpoint({"title": {"ver": 1, "seq": 8}}) == {}  # type: ignore[dict-item]
    assert registry.view_checkpoint(rows, ["title"]) == {"title": "Hello world"}


# ---------------------------------------------------------------------------
# hydrate.
# ---------------------------------------------------------------------------


def test_hydrate_installs_cells_and_later_drive_advances_once() -> None:
    """Hydrated cells are reused; later events advance them exactly once."""
    ctx, registry = _registry()
    session = _session_with_golden(ctx)
    events = list(session.events)
    rows = registry.checkpoint(session)
    # Checkpoint materializes cells at seq 10; drop them to simulate cold.
    for registration in registry._registrations.values():
        registration.cells.pop(session, None)
    snapshot = registry.hydrate(session, rows, events, 0)
    assert snapshot["as_of_seq"] == 10
    assert snapshot["values"]["title"] == "Hello world"
    assert registry.cached_snapshot(session) == snapshot
    # A later publication advances the hydrated cell exactly once.
    event = session.append(
        "session/title",
        {"title": "Renamed", "messageSeqs": [], "source": {"kind": "user"}},
    )
    ctx.emit("session/event", session, event)
    assert registry.snapshot(session)["values"]["title"] == "Renamed"


def test_hydrate_fast_path_uses_live_cells_without_refolding() -> None:
    """When every cell is already at the cut, hydrate refolds nothing."""
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    definition, calls = _counting_definition()
    registry.register(definition)
    registry.register(TITLE_INPUT_PROJECTION)
    session = Session.create("s1", seed=[], header=dict(HEADER))
    session.append("approval/policy", {"policy": "x"})
    session.append("approval/policy", {"policy": "y"})
    calls.clear()
    registry.snapshot(session)
    assert calls == [0, 1, 2]
    calls.clear()
    snapshot = registry.hydrate(session, {}, list(session.events), 0)
    assert snapshot == {"as_of_seq": 2, "values": {"counter": 3}}
    assert calls == []


def test_hydrate_skips_newer_live_cells() -> None:
    """A live cell newer than the restored row wins and is kept."""
    ctx, registry = _registry()
    session = _session_with_golden(ctx)
    _drive_all(ctx, session)
    assert registry.cached_snapshot(session)["as_of_seq"] == 10
    events = list(session.events[:6])
    stale_rows: dict[str, CheckpointRow] = {
        "title": CheckpointRow(ver=1, seq=5, val=None),
    }
    snapshot = registry.hydrate(session, stale_rows, events, 0)
    assert snapshot["as_of_seq"] == 5
    # The live cells (observed 10) were not replaced by the older rows.
    assert registry.cached_snapshot(session)["as_of_seq"] == 10


def test_hydrate_empty_events() -> None:
    """Hydrating an empty tail installs init states at seq -1."""
    _, registry = _registry()
    session = Session.create("s1", seed=[], header=dict(HEADER))
    snapshot = registry.hydrate(session, {}, [], 0)
    assert snapshot["as_of_seq"] == -1
    assert snapshot["values"]["title"] is None
    assert registry.cached_snapshot(session)["as_of_seq"] == -1


# ---------------------------------------------------------------------------
# Change feed.
# ---------------------------------------------------------------------------


def test_change_feed_fires_once_per_view_change() -> None:
    """Listeners hear each view-changing event once, never draft-only ones."""
    ctx, registry = _registry()
    seen: list[tuple[str, Any, int]] = []
    disposer = registry.on_changed(
        lambda session, key, value, seq: seen.append((key, value, seq))
    )
    session = _session_with_golden(ctx)
    _drive_all(ctx, session)
    title_events = [item for item in seen if item[0] == "title"]
    assert title_events == [("title", "Hello world", 8)]
    wd_events = [item for item in seen if item[0] == "workingDirectory"]
    assert wd_events == [("workingDirectory", "/tmp/proj", 9)]
    # Stats fires when its totals change (message, step/end); the outline
    # fires at its boundaries (turn/start, prompt, response) — never for a
    # draft-only assistant message.
    outline_seqs = sorted({item[2] for item in seen if item[0] == "turnOutline"})
    assert outline_seqs == [0, 1, 7]
    stats_seqs = sorted({item[2] for item in seen if item[0] == "sessionStats"})
    assert stats_seqs == [2, 3, 4, 5, 6]
    disposer()
    seen.clear()
    ctx.emit(
        "session/event",
        session,
        _event(
            "session/title",
            {"title": "Again", "messageSeqs": [], "source": {"kind": "user"}},
            11,
            3000,
        ),
    )
    assert seen == []


def test_change_feed_draft_only_message_is_quiet() -> None:
    """A draft-only assistant message does not notify (same turns identity)."""
    ctx, registry = _registry()
    seen: list[tuple[str, Any, int]] = []
    registry.on_changed(lambda session, key, value, seq: seen.append((key, value, seq)))
    session = Session.create("s1", seed=[], header=dict(HEADER))
    ctx.emit("session/event", session, _event("turn/start", {"turn": 1}, 0, 1000))
    seen.clear()
    message = _event(
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "a1",
                "role": "assistant",
                "content": [{"type": "text", "text": "draft"}],
            },
        },
        1,
        1100,
        surface_op="append",
    )
    ctx.emit("session/event", session, message)
    assert [item for item in seen if item[0] == "turnOutline"] == []
    ctx.emit("session/event", session, _event("turn/end", {"turn": 1}, 2, 1200))
    assert [item for item in seen if item[0] == "turnOutline"] == [
        (
            "turnOutline",
            [{"turn": 1, "seq": 0, "prompt": "", "response": "draft"}],
            2,
        )
    ]


# ---------------------------------------------------------------------------
# Drive: late build, double publication, stalls.
# ---------------------------------------------------------------------------


def test_drive_late_build_folds_history_before_first_event() -> None:
    """A unit registered after events flowed folds the prefix, then drives."""
    ctx, registry = _registry()
    session = _session_with_golden(ctx)
    _drive_all(ctx, session)
    definition, calls = _counting_definition()
    registry.register(definition)
    calls.clear()
    event = session.append("approval/policy", {"policy": "late"})
    ctx.emit("session/event", session, event)
    # Late build folded the 11 existing events, then the new one.
    assert calls == list(range(12))
    assert registry.snapshot(session)["values"]["counter"] == 12


def test_drive_skips_double_publication() -> None:
    """Publishing the same event twice folds it once."""
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    definition, calls = _counting_definition()
    registry.register(definition)
    session = Session.create("s1", seed=[], header=dict(HEADER))
    event = session.append("approval/policy", {"policy": "x"})
    ctx.emit("session/event", session, event)
    ctx.emit("session/event", session, event)
    # The constructor marker (seq 0) folded once on the late build, then
    # the published event folded once; the second publication was skipped.
    assert calls == [0, event.seq]
    assert registry.snapshot(session)["values"]["counter"] == 2


def test_drive_advances_a_gap_then_applies() -> None:
    """Events appended without publication are folded on the next drive."""
    ctx, registry = _registry()
    session = Session.create("s1", seed=[], header=dict(HEADER))
    ctx.emit("session/event", session, session.events[0])
    session.append("turn/start", {"turn": 1})
    session.append("turn/end", {"turn": 1})
    event = session.append(
        "session/title",
        {"title": "Gap", "messageSeqs": [], "source": {"kind": "user"}},
    )
    ctx.emit("session/event", session, event)
    assert registry.snapshot(session)["values"]["title"] == "Gap"
    assert registry.snapshot(session)["values"]["turnOutline"] == [
        {"turn": 1, "seq": 1, "prompt": "", "response": ""}
    ]


class _StubSession:
    """A minimal session-shaped stand-in (plain classes support weakrefs)."""

    def __init__(self, events: list[SessionEvent], seq: int) -> None:
        self.header = dict(HEADER)
        self.inherited_event_count = 0
        self.events = list(events)
        self.seq = seq


def test_advance_across_missing_seq_raises() -> None:
    """Advancing a cell across a gap or a mismatched seq raises loudly."""
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    definition, _ = _counting_definition()
    handle = registry.register(definition)
    events = [
        _event("approval/policy", {}, 0, 1000),
        _event("approval/policy", {}, 1, 1100),
    ]
    stub = _StubSession(events, 2)
    assert handle.state_of(stub) == 2
    # A later event claims seq 3 at index 2: the advance must refuse.
    stub.events.append(_event("approval/policy", {}, 3, 1200))
    stub.seq = 3
    with pytest.raises(ValueError, match="missing seq 2"):
        handle.state_of(stub)
    # A missing event (log shorter than the cursor) refuses the same way.
    stub.events.pop()
    stub.seq = 5
    with pytest.raises(ValueError, match="missing seq 2"):
        handle.state_of(stub)


def test_drive_stall_leaves_cell_behind() -> None:
    """A drive that cannot advance leaves the cell at its old watermark."""
    ctx, registry = _registry()
    session = Session.create("s1", seed=[], header=dict(HEADER))
    _drive_all(ctx, session)
    assert registry.cached_snapshot(session)["as_of_seq"] == 0
    # Forge a gap: the log's second event claims a later seq.
    forged = _event("approval/policy", {"policy": "x"}, 5, 2000)
    session.events.append(forged)
    ctx.emit("session/event", session, forged)
    assert registry.cached_snapshot(session)["as_of_seq"] == 0


def test_session_created_preseeds_empty_sessions_only() -> None:
    """session/created pre-seeds cells only for a genuinely empty log."""
    ctx, registry = _registry()
    empty = Session.from_restore("s1", [], dict(HEADER), 0, "detached")
    assert empty.seq == 0
    ctx.emit("session/created", empty)
    cached = registry.cached_snapshot(empty)
    assert cached is not None
    assert cached["as_of_seq"] == -1
    assert cached["values"]["title"] is None
    # A second created emission is a no-op (cells already exist).
    ctx.emit("session/created", empty)
    assert registry.cached_snapshot(empty)["as_of_seq"] == -1
    # A session with events is never pre-seeded; its first read folds lazily.
    full = _session_with_golden(ctx)
    ctx.emit("session/created", full)
    assert registry.cached_snapshot(full) is None
    assert registry.snapshot(full)["values"]["title"] == "Hello world"


def test_registry_dispose_unregisters_listeners() -> None:
    """After dispose, publications no longer drive any unit."""
    ctx, registry = _registry()
    registry.dispose()
    registry.dispose()
    session = _session_with_golden(ctx)
    _drive_all(ctx, session)
    assert registry.cached_snapshot(session) is None
    assert registry.snapshot(session)["values"]["title"] == "Hello world"


def test_registry_dispose_tolerates_non_callable_disposers() -> None:
    """Dispose skips listeners whose disposer is not callable."""
    registry = SessionProjectionRegistry(Context())
    registry._created_disposer = None  # type: ignore[assignment]
    registry._event_disposer = None  # type: ignore[assignment]
    registry.dispose()
    registry.dispose()


def test_unit_cell_and_checkpoint_row_shapes() -> None:
    """UnitCell carries the watermark and the two-slot view buffer."""
    cell = UnitCell(state={"a": 1}, observed_seq=3, views=[None, None])
    assert cell.state == {"a": 1}
    assert cell.observed_seq == 3
    assert cell.views == [None, None]
    row = CheckpointRow(ver=1, seq=3, val={"a": 1})
    assert row == {"ver": 1, "seq": 3, "val": {"a": 1}}
