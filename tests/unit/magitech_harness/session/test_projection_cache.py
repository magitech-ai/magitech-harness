"""Unit tests for the persisted projection cache."""

import dataclasses
import json
import logging
import math
import stat
import time
from pathlib import Path
from typing import Any

import pytest
from anima import Context

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.persistence_jsonl import encode_segment
from magitech_harness.session.projection_cache import (
    ProjectionCacheConfig,
    SessionProjectionCache,
    current_lifecycle_matches,
    identity_matches,
    identity_of,
    lifecycle_identity_of,
    predecessor_identity_matches,
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
from magitech_harness.session.store import Session

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "s1",
    "createdAt": 1700000000000,
    "cwd": "/home/malcom/work",
    "isSeeded": False,
    "delegationDepth": 0,
}

GOLDEN_VALUES: dict[str, Any] = {
    "title": "Hello world",
    "sessionStats": {
        "turns": 1,
        "steps": 1,
        "llm_ms": 300.0,
        "tool_ms": 200.0,
        "ttft_ms": 100.0,
        "ttft_steps": 1,
        "decode_ms": 200.0,
        "decode_tokens": 5,
    },
    "turnOutline": [
        {"turn": 1, "seq": 0, "prompt": "Hello world", "response": "Hi there"}
    ],
    "workingDirectory": "/tmp/proj",
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


def _session_with(events: list[SessionEvent]) -> Session:
    """A detached session holding exactly ``events`` (no constructor marker)."""
    return Session.from_restore("s1", list(events), dict(HEADER), 0, "detached")


def _golden_session() -> Session:
    """The Scenario A log as a detached session (exactly 10 events, seqs 0..9)."""
    return _session_with(_golden_events())


def _registry() -> tuple[Context, SessionProjectionRegistry]:
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    for definition in ALL_DEFINITIONS:
        registry.register(definition)
    return ctx, registry


def _cache(
    root: Path,
    registry: SessionProjectionRegistry,
    *,
    write_every_events: int = 200,
    write_interval_ms: int = 5000,
    flush_log: Any = None,
) -> SessionProjectionCache:
    return SessionProjectionCache(
        registry,
        root,
        ProjectionCacheConfig(write_every_events, write_interval_ms),
        flush_log=flush_log,
    )


def _cache_with_listeners(
    root: Path,
    registry: SessionProjectionRegistry,
    ctx: Context,
    **kwargs: Any,
) -> SessionProjectionCache:
    cache = _cache(root, registry, **kwargs)
    cache.install(ctx)
    return cache


def _record_path(root: Path, session_id: str = "s1") -> Path:
    return (
        root / "session_projcache" / "sessions" / f"{encode_segment(session_id)}.json"
    )


def _write_document(root: Path, text: str, session_id: str = "s1") -> Path:
    path = _record_path(root, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _envelope(record: dict[str, Any], version: int = 7) -> str:
    return json.dumps({"version": version, "record": record}, indent=2) + "\n"


def _golden_rows(seq: int = 9) -> dict[str, Any]:
    """Checkpoint rows for the golden log, as the registry folds them."""
    _ctx, registry = _registry()
    session = _golden_session()
    rows = registry.checkpoint(session)
    for row in rows.values():
        row["seq"] = seq
    return rows


def _valid_record(
    *,
    identity: dict[str, Any] | None = None,
    rows: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "identity": identity if identity is not None else identity_of(dict(HEADER), 0),
        "rows": rows if rows is not None else _golden_rows(),
    }


def _read_record(root: Path, session_id: str = "s1") -> dict[str, Any]:
    return json.loads(_record_path(root, session_id).read_text(encoding="utf-8"))


def _row_seqs(root: Path, session_id: str = "s1") -> dict[str, int]:
    record = _read_record(root, session_id)
    return {key: row["seq"] for key, row in record["record"]["rows"].items()}


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "200", None])
def test_config_rejects_bad_write_every_events(bad: Any) -> None:
    """The count throttle is a required integer >= 1."""
    with pytest.raises(ValueError, match="write_every_events"):
        ProjectionCacheConfig(bad, 5000)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "5000", None])
def test_config_rejects_bad_write_interval_ms(bad: Any) -> None:
    """The interval throttle is a required integer >= 1."""
    with pytest.raises(ValueError, match="write_interval_ms"):
        ProjectionCacheConfig(200, bad)


def test_config_accepts_shipped_values() -> None:
    """The shipped 200 events / 5000 ms construct cleanly."""
    config = ProjectionCacheConfig(200, 5000)
    assert config.write_every_events == 200
    assert config.write_interval_ms == 5000


# ---------------------------------------------------------------------------
# Identity helpers.
# ---------------------------------------------------------------------------


def test_identity_of_builds_complete_identity() -> None:
    """The fold identity carries the lifecycle fields plus the exact cut."""
    assert identity_of(dict(HEADER), 0) == {
        "format_version": 4,
        "created_at": 1700000000000,
        "cwd": "/home/malcom/work",
        "is_seeded": False,
        "inherited_event_count": 0,
    }
    seeded = {**HEADER, "isSeeded": True}
    assert identity_of(seeded, 3)["inherited_event_count"] == 3
    assert identity_of(seeded, 3)["is_seeded"] is True


def test_identity_of_omits_absent_cwd() -> None:
    """A header without cwd produces an identity without the key (never null)."""
    header = {key: value for key, value in HEADER.items() if key != "cwd"}
    identity = identity_of(header, 0)
    assert "cwd" not in identity
    assert lifecycle_identity_of(header) == {
        "format_version": 4,
        "created_at": 1700000000000,
        "is_seeded": False,
    }


@pytest.mark.parametrize("bad", [-1, True, 1.5, "0", None])
def test_identity_of_rejects_bad_cut(bad: Any) -> None:
    """The inherited cut must be an integer >= 0."""
    with pytest.raises(ValueError, match="inherited event count"):
        identity_of(dict(HEADER), bad)


def test_identity_of_rejects_unseeded_nonzero_cut() -> None:
    """An unseeded header with a nonzero cut is not an identity."""
    with pytest.raises(ValueError, match="unseeded"):
        identity_of(dict(HEADER), 1)


@pytest.mark.parametrize(
    ("stored", "expected", "matched"),
    [
        (
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            True,
        ),
        # An absent format version never matches: no current log can prove
        # that record's fold semantics.
        (
            {"created_at": 1, "is_seeded": False, "inherited_event_count": 0},
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            False,
        ),
        (
            {
                "format_version": 3,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            False,
        ),
        (
            {
                "format_version": 4,
                "created_at": 2,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            False,
        ),
        # Absent cwd matches only absent cwd.
        (
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            {
                "format_version": 4,
                "created_at": 1,
                "cwd": "/x",
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            False,
        ),
        (
            {
                "format_version": 4,
                "created_at": 1,
                "cwd": "/x",
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            False,
        ),
        # An absent is_seeded reads as unseeded.
        (
            {"format_version": 4, "created_at": 1, "inherited_event_count": 0},
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            True,
        ),
        (
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": True,
                "inherited_event_count": 0,
            },
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            False,
        ),
        # An absent cut reads as the unseeded lineage.
        (
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 0,
            },
            True,
        ),
        (
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 1,
            },
            False,
        ),
        (
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 1,
            },
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 1,
            },
            True,
        ),
        (
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 2,
            },
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 1,
            },
            False,
        ),
    ],
)
def test_identity_matches_matrix(
    stored: dict[str, Any], expected: dict[str, Any], matched: bool
) -> None:
    """The fold face matches lifecycle plus cut, with the absent-field rules."""
    assert identity_matches(stored, expected) is matched


@pytest.mark.parametrize(
    ("stored", "expected", "matched"),
    [
        (
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            True,
        ),
        # The listing face does not check the cut.
        (
            {
                "format_version": 4,
                "created_at": 1,
                "is_seeded": False,
                "inherited_event_count": 5,
            },
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            True,
        ),
        (
            {"created_at": 1, "is_seeded": False},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            False,
        ),
        (
            {"format_version": 4, "created_at": 2, "is_seeded": False},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            False,
        ),
    ],
)
def test_current_lifecycle_matches_matrix(
    stored: dict[str, Any], expected: dict[str, Any], matched: bool
) -> None:
    """The listing face matches lifecycle only; the cut is not consulted."""
    assert current_lifecycle_matches(stored, expected) is matched


@pytest.mark.parametrize(
    ("stored", "expected", "matched"),
    [
        # An absent format version is a predecessor (it predates the field).
        (
            {"created_at": 1, "is_seeded": False},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            True,
        ),
        (
            {"format_version": 3, "created_at": 1, "is_seeded": False},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            True,
        ),
        (
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            False,
        ),
        (
            {"format_version": 5, "created_at": 1, "is_seeded": False},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            False,
        ),
        (
            {"format_version": 3, "created_at": 2, "is_seeded": False},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            False,
        ),
        (
            {"format_version": 3, "created_at": 1, "is_seeded": True},
            {"format_version": 4, "created_at": 1, "is_seeded": False},
            False,
        ),
    ],
)
def test_predecessor_identity_matches_matrix(
    stored: dict[str, Any], expected: dict[str, Any], matched: bool
) -> None:
    """The predecessor face matches older (or absent) versions with same lifecycle."""
    assert predecessor_identity_matches(stored, expected) is matched


# ---------------------------------------------------------------------------
# Open: envelope stamps, schema validation, backup-and-skip.
# ---------------------------------------------------------------------------


def test_open_with_no_cache_dir_is_empty(tmp_path: Path) -> None:
    """A root with no cache directory opens as an empty cache."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    assert cache.cached_snapshot(dict(HEADER)) is None


@pytest.mark.parametrize("version", [3, 4, 5, 6, 7])
def test_open_accepts_compatible_envelope_stamps(tmp_path: Path, version: int) -> None:
    """Stamps 7, 3, 4, 5, 6 are accepted; the record serves reads."""
    _write_document(tmp_path, _envelope(_valid_record(), version))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    record = cache.record_for("s1", identity_of(dict(HEADER), 0))
    assert record is not None
    assert record["rows"]["title"]["val"] == "Hello world"


@pytest.mark.parametrize("text", ["not json at all", "[1, 2]", '"just a string"'])
def test_foreign_documents_read_as_absent_silently(tmp_path: Path, text: str) -> None:
    """Malformed JSON and non-object documents are foreign: no backup, no log."""
    path = _write_document(tmp_path, text)
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    # The file is untouched: foreign reads as absent, silently.
    assert path.read_text(encoding="utf-8") == text
    assert list(path.parent.glob("*.bak.*")) == []


@pytest.mark.parametrize("version", [2, 8, 0, -1])
def test_unaccepted_stamp_reads_as_absent_silently(
    tmp_path: Path, version: int
) -> None:
    """A stamp outside {7, 3, 4, 5, 6} is foreign: no backup, no log."""
    text = _envelope(_valid_record(), version)
    path = _write_document(tmp_path, text)
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    assert path.read_text(encoding="utf-8") == text
    assert list(path.parent.glob("*.bak.*")) == []


def test_missing_record_key_is_schema_invalid(tmp_path: Path) -> None:
    """An accepted stamp without a record is schema-invalid, so it is backed up."""
    path = _write_document(tmp_path, json.dumps({"version": 7}, indent=2) + "\n")
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    assert not path.exists()
    backups = list(path.parent.glob("s1.json.bak.*"))
    assert len(backups) == 1
    assert (
        backups[0].read_text(encoding="utf-8")
        == json.dumps({"version": 7}, indent=2) + "\n"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda record: record["identity"].pop("created_at"),
        lambda record: record["identity"].update(created_at=-1),
        lambda record: record["identity"].update(created_at=True),
        lambda record: record["identity"].update(created_at="1700000000000"),
        lambda record: record["identity"].update(format_version=-1),
        lambda record: record["identity"].update(cwd=12),
        lambda record: record["identity"].update(is_seeded="yes"),
        lambda record: record["identity"].update(inherited_event_count=-1),
        lambda record: record.pop("identity"),
        lambda record: record.update(rows=[]),
        lambda record: record["rows"].update(title={"ver": -1, "seq": 9, "val": "x"}),
        lambda record: record["rows"].update(title={"ver": True, "seq": 9, "val": "x"}),
        lambda record: record["rows"].update(title={"ver": 1, "seq": -2, "val": "x"}),
        lambda record: record["rows"].update(title={"ver": 1, "seq": 9}),
        lambda record: record["rows"].update(title="not a row"),
    ],
)
def test_schema_invalid_record_is_backed_up_and_skipped(
    tmp_path: Path, caplog: Any, mutate: Any
) -> None:
    """A schema-invalid record is moved aside, logged, and served as uncached."""
    record = _valid_record()
    mutate(record)
    path = _write_document(tmp_path, _envelope(record))
    original = path.read_text(encoding="utf-8")
    _, registry = _registry()
    with caplog.at_level(logging.ERROR, logger="harness.session"):
        cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    assert not path.exists()
    backups = list(path.parent.glob("s1.json.bak.*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == original
    assert "failed schema validation" in caplog.text
    assert "moved to" in caplog.text
    assert "treated as absent" in caplog.text


def test_schema_invalid_val_is_backed_up(tmp_path: Path) -> None:
    """A row whose val is not losslessly JSON-serializable is schema-invalid."""
    text = (
        '{"version": 7, "record": {"identity": '
        '{"format_version": 4, "created_at": 1700000000000, "is_seeded": false, '
        '"inherited_event_count": 0}, "rows": '
        '{"title": {"ver": 1, "seq": 9, "val": NaN}}}}'
    )
    path = _write_document(tmp_path, text)
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    assert not path.exists()
    assert len(list(path.parent.glob("s1.json.bak.*"))) == 1


def test_non_record_json_is_schema_invalid(tmp_path: Path) -> None:
    """An accepted stamp whose record is not an object is backed up, not foreign."""
    path = _write_document(
        tmp_path, json.dumps({"version": 7, "record": [1, 2]}) + "\n"
    )
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    assert not path.exists()
    assert len(list(path.parent.glob("s1.json.bak.*"))) == 1


def test_backup_stamp_is_local_minute_resolution(tmp_path: Path) -> None:
    """Backups land as <id>.json.bak.<YYYYMMDDHHmm> in the same directory."""
    record = _valid_record()
    record["identity"].pop("created_at")
    path = _write_document(tmp_path, _envelope(record))
    _, registry = _registry()
    _cache(tmp_path, registry)
    backups = list(path.parent.glob("s1.json.bak.*"))
    assert len(backups) == 1
    stamp = backups[0].name.removeprefix("s1.json.bak.")
    assert len(stamp) == 12 and stamp.isdigit()
    assert stamp == time.strftime("%Y%m%d%H%M", time.localtime())


def test_same_minute_backup_overwrites_previous(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A same-minute backup of the same key keeps the newer bytes only."""
    import magitech_harness.session.projection_cache as cache_module

    monkeypatch.setattr(cache_module, "_backup_stamp", lambda: "202611131200")
    first = _valid_record()
    first["identity"].pop("created_at")
    path = _write_document(tmp_path, _envelope(first))
    _, registry = _registry()
    _cache(tmp_path, registry)
    second = _valid_record()
    second["identity"].update(created_at="bad")
    _write_document(tmp_path, _envelope(second))
    _cache(tmp_path, registry)
    backups = list(path.parent.glob("s1.json.bak.*"))
    assert len(backups) == 1
    assert backups[0].name == "s1.json.bak.202611131200"
    assert json.loads(backups[0].read_text(encoding="utf-8"))["record"] == second


def test_backup_failure_fails_open_loud(tmp_path: Path, monkeypatch: Any) -> None:
    """When the document cannot be moved aside, open fails loud instead."""
    import magitech_harness.session.projection_cache as cache_module

    monkeypatch.setattr(cache_module, "_backup_stamp", lambda: "202611131200")
    record = _valid_record()
    record["identity"].pop("created_at")
    path = _write_document(tmp_path, _envelope(record))
    # A directory at the backup destination makes the rename fail.
    (path.parent / "s1.json.bak.202611131200").mkdir()
    _, registry = _registry()
    with pytest.raises(OSError):
        _cache(tmp_path, registry)
    # The loud path leaves the corrupt document in place.
    assert path.exists()


def test_open_ignores_stray_files_and_directories(tmp_path: Path) -> None:
    """Non-record files, temp files, and directories never disturb the open."""
    sessions_dir = tmp_path / "session_projcache" / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "notes.txt").write_text("not a record", encoding="utf-8")
    (sessions_dir / "s1.json.tmp").write_text("{}", encoding="utf-8")
    (sessions_dir / "weird.json").mkdir()
    (sessions_dir / "s1.json.bak.202611131200").write_text(
        "old backup bytes", encoding="utf-8"
    )
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None


def test_open_ignores_legacy_whole_unit_file(tmp_path: Path) -> None:
    """A stray <root>/session_projcache.json file is never read."""
    (tmp_path / "session_projcache.json").write_text(
        "not a cache record", encoding="utf-8"
    )
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None


def test_unreadable_document_reads_as_absent(tmp_path: Path, monkeypatch: Any) -> None:
    """A document that cannot be read is foreign, never fatal."""
    path = _write_document(tmp_path, _envelope(_valid_record()))

    def _boom(self: Path, *args: Any, **kwargs: Any) -> str:
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "read_text", _boom)
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    assert path.exists()


# ---------------------------------------------------------------------------
# put: envelope, modes, atomic replacement, validation.
# ---------------------------------------------------------------------------


def test_put_writes_versioned_pretty_envelope(tmp_path: Path) -> None:
    """A put lands as {"version": 7, "record": ...}, indented, newline-terminated."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    rows = registry.checkpoint(session)
    cache.put("s1", identity_of(dict(HEADER), 0), rows)
    path = _record_path(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.startswith('{\n  "version": 7,\n  "record": {')
    document = json.loads(text)
    assert document["version"] == 7
    assert document["record"]["identity"] == identity_of(dict(HEADER), 0)
    assert document["record"]["rows"]["title"]["val"] == "Hello world"
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is not None


def test_put_sets_file_and_directory_modes(tmp_path: Path) -> None:
    """Record documents are 0o600; their directories are 0o700."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.put("s1", identity_of(dict(HEADER), 0), registry.checkpoint(session))
    path = _record_path(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700


def test_put_replaces_the_whole_record(tmp_path: Path) -> None:
    """Each put replaces the record wholesale; no stale row survives."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.put("s1", identity_of(dict(HEADER), 0), registry.checkpoint(session))
    assert _read_record(tmp_path)["record"]["rows"]["title"]["val"] == "Hello world"
    rows = {"title": {"ver": 1, "seq": 3, "val": "Renamed"}}
    cache.put("s1", identity_of(dict(HEADER), 0), rows)
    record = _read_record(tmp_path)["record"]
    assert record["rows"] == rows
    assert "sessionStats" not in record["rows"]


def test_put_encodes_unsafe_session_ids(tmp_path: Path) -> None:
    """Session ids that are not path-safe are encoded, never traversed."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.put("../evil", identity_of(dict(HEADER), 0), registry.checkpoint(session))
    assert not (tmp_path / "evil.json").exists()
    assert not (tmp_path.parent / "evil.json").exists()
    encoded = _record_path(tmp_path, "../evil")
    assert encoded.exists()
    assert cache.record_for("../evil", identity_of(dict(HEADER), 0)) is not None


def test_put_rejects_empty_session_id(tmp_path: Path) -> None:
    """An empty id cannot be encoded, so the put refuses loudly."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    with pytest.raises(ValueError, match="empty path segment"):
        cache.put("", identity_of(dict(HEADER), 0), {})


def test_put_rejects_non_json_checkpoint(tmp_path: Path) -> None:
    """A checkpoint that is not losslessly JSON-serializable refuses loudly."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    rows = {"title": {"ver": 1, "seq": 0, "val": {"nested": {1, 2, 3}}}}
    with pytest.raises(TypeError, match="not losslessly JSON-serializable"):
        cache.put("s1", identity_of(dict(HEADER), 0), rows)  # type: ignore[arg-type]
    assert not _record_path(tmp_path).exists()
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None


def test_put_rejects_non_finite_checkpoint_values(tmp_path: Path) -> None:
    """NaN and infinite floats are not lossless JSON, so the put refuses."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    for bad in (math.nan, math.inf, -math.inf):
        rows = {"title": {"ver": 1, "seq": 0, "val": bad}}
        with pytest.raises(TypeError, match="not losslessly JSON-serializable"):
            cache.put("s1", identity_of(dict(HEADER), 0), rows)
    assert not _record_path(tmp_path).exists()


def test_put_rejects_schema_invalid_identity(tmp_path: Path) -> None:
    """A record whose identity fails the schema refuses loudly, unwritten."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    identity = {"format_version": 4, "is_seeded": False, "inherited_event_count": 0}
    with pytest.raises(ValueError, match="failed schema validation"):
        cache.put("s1", identity, {"title": {"ver": 1, "seq": 0, "val": None}})
    assert not _record_path(tmp_path).exists()


def test_put_failure_at_rename_leaves_no_temp_file(tmp_path: Path) -> None:
    """A failure between temp-write and rename removes the temp and raises."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    # A directory at the target path makes the atomic rename fail.
    target = _record_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    with pytest.raises(OSError):
        cache.put("s1", identity_of(dict(HEADER), 0), {})
    assert list(target.parent.glob(".*.tmp")) == []
    assert target.is_dir()


def test_put_failure_at_open_propagates_without_temp_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A failure creating the temp file propagates and leaves nothing behind."""
    import magitech_harness.session.projection_cache as cache_module

    sessions_dir = tmp_path / "session_projcache" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / ".deadbeef.tmp").write_text("stale temp", encoding="utf-8")
    monkeypatch.setattr(cache_module, "uuid4", lambda: _FixedHex("deadbeef"))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    with pytest.raises(FileExistsError):
        cache.put("s1", identity_of(dict(HEADER), 0), {})
    assert (sessions_dir / ".deadbeef.tmp").read_text(encoding="utf-8") == "stale temp"


class _FixedHex:
    """A stand-in uuid whose hex collides with a pre-created temp file."""

    def __init__(self, value: str) -> None:
        self.hex = value


def test_put_after_failed_put_recovers(tmp_path: Path) -> None:
    """A failed put leaves no record; the next good put lands cleanly."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    target = _record_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    with pytest.raises(OSError):
        cache.put("s1", identity_of(dict(HEADER), 0), {})
    target.rmdir()
    session = _golden_session()
    cache.put("s1", identity_of(dict(HEADER), 0), registry.checkpoint(session))
    assert _read_record(tmp_path)["record"]["rows"]["title"]["val"] == "Hello world"
    assert list(target.parent.glob(".*.tmp")) == []


# ---------------------------------------------------------------------------
# Read faces: record_for, cached_snapshot, predecessor title, hydrate, cold.
# ---------------------------------------------------------------------------


def test_record_for_returns_none_without_a_record(tmp_path: Path) -> None:
    """No stored record reads as absent."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None


def test_record_for_rejects_identity_mismatch_and_keeps_the_file(
    tmp_path: Path,
) -> None:
    """A mismatched record is treated as absent, never deleted or repaired."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.put("s1", identity_of(dict(HEADER), 0), registry.checkpoint(session))
    changed = {**HEADER, "createdAt": 1700000000001}
    assert cache.record_for("s1", identity_of(changed, 0)) is None
    assert _record_path(tmp_path).exists()
    # The stored record still serves its own lifecycle.
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is not None


def test_cached_snapshot_serves_stored_rows(tmp_path: Path) -> None:
    """The listing face views stored rows at the lowest served watermark."""
    _write_document(tmp_path, _envelope(_valid_record()))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    snapshot = cache.cached_snapshot(dict(HEADER))
    assert snapshot is not None
    assert snapshot["as_of_seq"] == 9
    assert snapshot["values"] == GOLDEN_VALUES


def test_cached_snapshot_keys_subset(tmp_path: Path) -> None:
    """A key selection narrows the served values (and the watermark)."""
    _write_document(tmp_path, _envelope(_valid_record()))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    snapshot = cache.cached_snapshot(dict(HEADER), ["title"])
    assert snapshot == {"as_of_seq": 9, "values": {"title": "Hello world"}}


def test_cached_snapshot_rejects_lifecycle_mismatch(tmp_path: Path) -> None:
    """A record bound to another lifecycle serves nothing to the listing."""
    _write_document(tmp_path, _envelope(_valid_record()))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    changed = {**HEADER, "cwd": "/elsewhere"}
    assert cache.cached_snapshot(changed) is None


def test_cached_snapshot_accepts_different_cut(tmp_path: Path) -> None:
    """The listing face does not check the inherited cut (no fold is seeded)."""
    identity = identity_of(dict(HEADER), 0)
    identity["inherited_event_count"] = 5
    _write_document(tmp_path, _envelope(_valid_record(identity=identity)))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    snapshot = cache.cached_snapshot(dict(HEADER))
    assert snapshot is not None
    assert snapshot["values"] == GOLDEN_VALUES
    # The fold face, by contrast, refuses the same record.
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None


def test_cached_snapshot_skips_version_mismatched_rows(tmp_path: Path) -> None:
    """Rows whose ver mismatches leave their key absent; the watermark follows."""
    rows = _golden_rows()
    rows["title"]["ver"] = 99
    _write_document(tmp_path, _envelope(_valid_record(rows=rows)))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    snapshot = cache.cached_snapshot(dict(HEADER))
    assert snapshot is not None
    assert "title" not in snapshot["values"]
    assert snapshot["values"]["workingDirectory"] == "/tmp/proj"
    assert snapshot["as_of_seq"] == 9


def test_cached_snapshot_returns_none_when_no_row_serves(tmp_path: Path) -> None:
    """When every row is unusable, the listing face serves nothing."""
    rows = _golden_rows()
    for row in rows.values():
        row["ver"] = 99
    _write_document(tmp_path, _envelope(_valid_record(rows=rows)))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.cached_snapshot(dict(HEADER)) is None


def test_cached_predecessor_title_serves_older_records(tmp_path: Path) -> None:
    """A predecessor record exposes its title as a listing hint."""
    identity = identity_of(dict(HEADER), 0)
    identity["format_version"] = 3
    _write_document(tmp_path, _envelope(_valid_record(identity=identity)))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    snapshot = cache.cached_predecessor_title(dict(HEADER))
    assert snapshot == {"as_of_seq": 9, "values": {"title": "Hello world"}}
    # The same record is not a current record: the fold face rejects it.
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None
    assert cache.cached_snapshot(dict(HEADER)) is None


def test_cached_predecessor_title_serves_versionless_records(
    tmp_path: Path,
) -> None:
    """A record without format_version is a predecessor for the title hint."""
    identity = identity_of(dict(HEADER), 0)
    identity.pop("format_version")
    _write_document(tmp_path, _envelope(_valid_record(identity=identity)))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    snapshot = cache.cached_predecessor_title(dict(HEADER))
    assert snapshot == {"as_of_seq": 9, "values": {"title": "Hello world"}}


@pytest.mark.parametrize("version", [4, 5])
def test_cached_predecessor_title_rejects_current_and_newer(
    tmp_path: Path, version: int
) -> None:
    """A current or newer record is not a predecessor."""
    identity = identity_of(dict(HEADER), 0)
    identity["format_version"] = version
    _write_document(tmp_path, _envelope(_valid_record(identity=identity)))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.cached_predecessor_title(dict(HEADER)) is None


def test_cached_predecessor_title_rejects_unrelated(tmp_path: Path) -> None:
    """A predecessor record from another lifecycle serves no title."""
    identity = identity_of({**HEADER, "createdAt": 1}, 0)
    identity["format_version"] = 3
    _write_document(tmp_path, _envelope(_valid_record(identity=identity)))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.cached_predecessor_title(dict(HEADER)) is None


def test_cached_predecessor_title_requires_title_row(tmp_path: Path) -> None:
    """A predecessor record whose title row is unusable serves nothing."""
    identity = identity_of(dict(HEADER), 0)
    identity["format_version"] = 3
    rows = _golden_rows()
    rows["title"]["ver"] = 99
    _write_document(tmp_path, _envelope(_valid_record(identity=identity, rows=rows)))
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.cached_predecessor_title(dict(HEADER)) is None


def test_hydrate_prepared_seeds_from_matching_record(tmp_path: Path) -> None:
    """Hydration installs the cached rows as live cells."""
    # Build the rows with one registry/session pair, then hydrate a fresh
    # pair: the fast path only applies when live cells already sit at the
    # cut, so reusing the same pair would never read the cached rows.
    _, builder = _registry()
    rows = builder.checkpoint(_golden_session())
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.put("s1", identity_of(dict(HEADER), 0), rows)
    snapshot = cache.hydrate_prepared(session, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}
    assert registry.cached_snapshot(session) == snapshot


def test_hydrate_prepared_without_a_record_folds_the_log(tmp_path: Path) -> None:
    """Without a record, hydration folds the full log."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    snapshot = cache.hydrate_prepared(session, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}


def test_hydrate_prepared_falls_back_on_a_stale_schema(tmp_path: Path) -> None:
    """Cached rows that fail the fold never make a valid session unreadable."""
    # Build the rows with one registry/session pair, then hydrate a fresh
    # pair: the fast path only applies when live cells already sit at the
    # cut, so reusing the same pair would never read the cached rows.
    _, builder = _registry()
    rows = builder.checkpoint(_golden_session())
    rows["title"]["val"] = 123  # Passes the record schema; fails the unit schema.
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.put("s1", identity_of(dict(HEADER), 0), rows)
    snapshot = cache.hydrate_prepared(session, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}


def test_cold_snapshot_refolds_and_writes_back(tmp_path: Path) -> None:
    """The first cold read refolds from the log and creates the cache row."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    snapshot = cache.cold_snapshot(dict(HEADER), 0, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}
    assert _read_record(tmp_path)["record"]["rows"]["title"]["val"] == "Hello world"
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is not None


def test_cold_snapshot_seeds_from_matching_record(tmp_path: Path) -> None:
    """A cold read seeds from the cached rows and refreshes the checkpoint."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.put("s1", identity_of(dict(HEADER), 0), registry.checkpoint(session))
    snapshot = cache.cold_snapshot(dict(HEADER), 0, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}
    assert _row_seqs(tmp_path)["title"] == 9


def test_cold_snapshot_refolds_on_identity_mismatch(tmp_path: Path) -> None:
    """An unrelated record never seeds a cold read; the write-back replaces it."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    old_identity = identity_of({**HEADER, "createdAt": 1}, 0)
    cache.put("s1", old_identity, registry.checkpoint(session))
    snapshot = cache.cold_snapshot(dict(HEADER), 0, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}
    record = _read_record(tmp_path)["record"]
    assert record["identity"]["created_at"] == 1700000000000


def test_cold_snapshot_write_back_failure_is_fail_soft(
    tmp_path: Path, caplog: Any
) -> None:
    """A failed write-back only costs a longer tail replay, never the read."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    target = _record_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    with caplog.at_level(logging.WARNING, logger="harness.session"):
        snapshot = cache.cold_snapshot(dict(HEADER), 0, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}
    assert "cold-read write-back" in caplog.text
    assert cache.record_for("s1", identity_of(dict(HEADER), 0)) is None


# ---------------------------------------------------------------------------
# Write points, throttles, fail-soft, listeners.
# ---------------------------------------------------------------------------


def _emit(ctx: Context, session: Session, events: list[SessionEvent]) -> None:
    for event in events:
        ctx.emit("session/event", session, event)


def test_create_point_writes_the_seed_cut(tmp_path: Path) -> None:
    """Session creation is the first mandatory write point."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(tmp_path, registry, ctx)
    session = Session.create("s1", seed=[], header=dict(HEADER))
    assert session.seq == 1  # one constructor marker
    ctx.emit("session/created", session)
    assert _row_seqs(tmp_path) == {
        "title": 0,
        "titleInput": 0,
        "sessionStats": 0,
        "turnOutline": 0,
        "workingDirectory": 0,
    }
    record = _read_record(tmp_path)["record"]
    assert record["identity"] == identity_of(dict(HEADER), 0)
    cache.dispose()


def test_turn_end_point_and_dirty_tail(tmp_path: Path) -> None:
    """turn/end writes; later events stay dirty until the detach write."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(tmp_path, registry, ctx)
    events = _golden_events()
    session = _session_with(events[:8])
    _emit(ctx, session, events[:8])
    assert _row_seqs(tmp_path)["title"] == 7
    # Append the tail; it stays dirty (no write) until disposal.
    title_event = session.append(
        "session/title",
        {"title": "Hello world", "messageSeqs": [1], "source": {"kind": "fallback"}},
    )
    ctx.emit("session/event", session, title_event)
    cwd_event = session.append("working-directory/change", {"cwd": "/tmp/proj"})
    ctx.emit("session/event", session, cwd_event)
    assert _row_seqs(tmp_path)["title"] == 7
    ctx.emit("session/disposed", session)
    assert _row_seqs(tmp_path) == {
        "title": 9,
        "titleInput": 9,
        "sessionStats": 9,
        "turnOutline": 9,
        "workingDirectory": 9,
    }
    cache.dispose()


def _append_policy(ctx: Context, session: Session) -> SessionEvent:
    """Append one fold-ignored event and publish it (grows the live cursor)."""
    event = session.append("approval/policy", {"policy": "x"})
    ctx.emit("session/event", session, event)
    return event


def test_count_threshold_writes(tmp_path: Path) -> None:
    """Every N committed events force a durable write between mandatory points."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(
        tmp_path, registry, ctx, write_every_events=3, write_interval_ms=5000
    )
    session = Session.create("s1", seed=[], header=dict(HEADER))
    _append_policy(ctx, session)
    _append_policy(ctx, session)
    assert not _record_path(tmp_path).exists()
    _append_policy(ctx, session)
    assert _row_seqs(tmp_path)["title"] == 3
    _append_policy(ctx, session)
    _append_policy(ctx, session)
    assert _row_seqs(tmp_path)["title"] == 3
    _append_policy(ctx, session)
    assert _row_seqs(tmp_path)["title"] == 6
    cache.dispose()


def test_interval_threshold_writes_once(tmp_path: Path) -> None:
    """The interval trigger fires once per dirty window, not once per event."""
    ctx, registry = _registry()
    calls: list[str] = []
    cache = _cache_with_listeners(
        tmp_path,
        registry,
        ctx,
        write_every_events=200,
        write_interval_ms=50,
        flush_log=lambda session: calls.append(session.session_id),
    )
    session = Session.create("s1", seed=[], header=dict(HEADER))
    _append_policy(ctx, session)
    _append_policy(ctx, session)
    time.sleep(0.3)
    assert calls == ["s1"]
    assert _row_seqs(tmp_path)["title"] == 2
    cache.dispose()


def test_dispose_cancels_armed_timer(tmp_path: Path) -> None:
    """Disposal cancels a pending interval timer; no late write lands."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(
        tmp_path, registry, ctx, write_every_events=200, write_interval_ms=50
    )
    events = _golden_events()
    session = _session_with(events[:4])
    _emit(ctx, session, events[:1])
    cache.dispose()
    time.sleep(0.3)
    assert not _record_path(tmp_path).exists()


def test_mark_clean_resets_pending_and_cancels_timer(tmp_path: Path) -> None:
    """mark_clean clears the counter and the timer, so the window restarts."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(
        tmp_path, registry, ctx, write_every_events=3, write_interval_ms=50
    )
    session = Session.create("s1", seed=[], header=dict(HEADER))
    _append_policy(ctx, session)
    cache.mark_clean(session)
    time.sleep(0.2)
    assert not _record_path(tmp_path).exists()
    _append_policy(ctx, session)
    _append_policy(ctx, session)
    assert not _record_path(tmp_path).exists()
    _append_policy(ctx, session)
    assert _row_seqs(tmp_path)["title"] == 4
    cache.dispose()


def test_mark_clean_without_dirty_state_is_a_noop(tmp_path: Path) -> None:
    """mark_clean on a session with no dirty entry does nothing."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    session = _golden_session()
    cache.mark_clean(session)
    assert not _record_path(tmp_path).exists()


def test_flush_soft_contains_write_failures(
    tmp_path: Path, caplog: Any, monkeypatch: Any
) -> None:
    """A failing turn/end write warns and leaves the session unaffected."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(tmp_path, registry, ctx)
    events = _golden_events()
    session = _session_with(events[:8])
    _emit(ctx, session, events[:8])
    assert _row_seqs(tmp_path)["title"] == 7
    monkeypatch.setattr(cache, "put", _raising_put)
    title_event = session.append(
        "session/title",
        {"title": "Hello world", "messageSeqs": [1], "source": {"kind": "fallback"}},
    )
    ctx.emit("session/event", session, title_event)
    cwd_event = session.append("working-directory/change", {"cwd": "/tmp/proj"})
    ctx.emit("session/event", session, cwd_event)
    tail_end = session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    with caplog.at_level(logging.WARNING, logger="harness.session"):
        ctx.emit("session/event", session, tail_end)
    assert 'turn/end write for "s1" failed' in caplog.text
    assert "(cache stays stale)" in caplog.text
    # The on-disk record is still the older cut; the session is unaffected.
    assert _row_seqs(tmp_path)["title"] == 7
    assert registry.snapshot(session)["values"]["title"] == "Hello world"
    monkeypatch.undo()
    # With put restored, the cold read refolds the tail and writes back.
    snapshot = cache.cold_snapshot(dict(HEADER), 0, list(session.events))
    assert snapshot == {"as_of_seq": 10, "values": GOLDEN_VALUES}
    assert _row_seqs(tmp_path)["title"] == 10
    cache.dispose()


def _raising_put(
    session_id: str, identity: dict[str, Any], rows: dict[str, Any]
) -> None:
    raise OSError("simulated crash between log flush and cache write")


def test_write_is_not_fail_soft(tmp_path: Path, monkeypatch: Any) -> None:
    """write() itself propagates failures; only flush_soft contains them."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(tmp_path, registry, ctx)
    session = _golden_session()
    monkeypatch.setattr(cache, "put", _raising_put)
    with pytest.raises(OSError, match="simulated crash"):
        cache.write(session)
    cache.dispose()


def test_write_flushes_log_before_cache_write(tmp_path: Path) -> None:
    """The log flush runs before the record lands (ordering, by injection)."""
    _, registry = _registry()
    order: list[str] = []
    session = _golden_session()
    cache = _cache(
        tmp_path,
        registry,
        flush_log=lambda _session: order.append("flush"),
    )
    original_put = cache.put

    def _recording_put(
        session_id: str, identity: dict[str, Any], rows: dict[str, Any]
    ) -> None:
        order.append("put")
        original_put(session_id, identity, rows)

    cache.put = _recording_put  # type: ignore[method-assign]
    cache.write(session)
    assert order == ["flush", "put"]


def test_write_without_flush_log_skips_the_flush(tmp_path: Path) -> None:
    """A cache with no log-flush seam still writes its record."""
    _, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.write(session)
    assert _row_seqs(tmp_path)["title"] == 9


def test_write_propagates_a_log_flush_failure(tmp_path: Path) -> None:
    """A failing log flush aborts the write before the record lands."""
    _, registry = _registry()
    session = _golden_session()

    def _failing_flush(_session: Session) -> None:
        raise OSError("log flush failed")

    cache = _cache(tmp_path, registry, flush_log=_failing_flush)
    with pytest.raises(OSError, match="log flush failed"):
        cache.write(session)
    assert not _record_path(tmp_path).exists()


def test_event_listener_never_raises(
    tmp_path: Path, caplog: Any, monkeypatch: Any
) -> None:
    """A failing flush inside the event listener warns instead of raising."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(tmp_path, registry, ctx)
    monkeypatch.setattr(cache, "flush_soft", _raising_flush_soft)
    events = _golden_events()
    session = _session_with(events[:8])
    with caplog.at_level(logging.WARNING, logger="harness.session"):
        _emit(ctx, session, events[:8])
    assert "event listener" in caplog.text
    assert registry.snapshot(session)["values"]["title"] is None
    cache.dispose()


def _raising_flush_soft(session: Session, trigger: str) -> None:
    raise RuntimeError("simulated listener failure")


def test_created_listener_never_raises(
    tmp_path: Path, caplog: Any, monkeypatch: Any
) -> None:
    """A failing flush inside the created listener warns instead of raising."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(tmp_path, registry, ctx)
    monkeypatch.setattr(cache, "flush_soft", _raising_flush_soft)
    session = Session.create("s1", seed=[], header=dict(HEADER))
    with caplog.at_level(logging.WARNING, logger="harness.session"):
        ctx.emit("session/created", session)
    assert "create listener" in caplog.text
    cache.dispose()


def test_disposed_listener_never_raises_and_drops_dirty(
    tmp_path: Path, caplog: Any, monkeypatch: Any
) -> None:
    """A failing detach flush warns, and the dirty entry is still dropped."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(
        tmp_path, registry, ctx, write_every_events=2, write_interval_ms=5000
    )
    events = _golden_events()
    session = _session_with(events)
    _emit(ctx, session, events[:1])
    monkeypatch.setattr(cache, "flush_soft", _raising_flush_soft)
    with caplog.at_level(logging.WARNING, logger="harness.session"):
        ctx.emit("session/disposed", session)
    assert "detach listener" in caplog.text
    monkeypatch.undo()
    # The dirty entry was dropped: one more event restarts the counter at 1.
    _emit(ctx, session, events[1:2])
    assert not _record_path(tmp_path).exists()
    cache.dispose()


def test_install_twice_raises(tmp_path: Path) -> None:
    """A second install while one is active refuses loudly."""
    ctx, registry = _registry()
    cache = _cache(tmp_path, registry)
    cache.install(ctx)
    with pytest.raises(ValueError, match="already installed"):
        cache.install(ctx)
    cache.dispose()


def test_dispose_is_idempotent_and_reinstall_works(tmp_path: Path) -> None:
    """Dispose unregisters once; a fresh install after disposal works."""
    ctx, registry = _registry()
    cache = _cache(tmp_path, registry)
    cache.dispose()
    cache.dispose()
    cache.install(ctx)
    events = _golden_events()
    session = _session_with(events[:8])
    _emit(ctx, session, events[:8])
    assert _row_seqs(tmp_path)["title"] == 7
    cache.dispose()
    cache.dispose()


# ---------------------------------------------------------------------------
# stateVersion mismatch (§5.2) and crash ordering (§5.5).
# ---------------------------------------------------------------------------


def test_state_version_mismatch_discards_row_and_refolds(tmp_path: Path) -> None:
    """A cached row at another ver is discarded; the log refolds it."""
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    handles = [registry.register(definition) for definition in ALL_DEFINITIONS]
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    cache.put("s1", identity_of(dict(HEADER), 0), registry.checkpoint(session))
    # Simulate a unit upgrade: dispose the old title registration, then
    # re-register the same key at state_version 2.
    handles[0].dispose()
    upgraded = dataclasses.replace(TITLE_PROJECTION, state_version=2)
    registry.register(upgraded)
    snapshot = cache.cold_snapshot(dict(HEADER), 0, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}
    assert _read_record(tmp_path)["record"]["rows"]["title"]["ver"] == 2


def test_restore_with_mismatched_row_at_base_zero_refolds(tmp_path: Path) -> None:
    """restore() discards a mismatched row over the full log without raising."""
    _ctx, registry = _registry()
    session = _golden_session()
    rows = registry.checkpoint(session)
    rows["title"]["ver"] = 99
    restored = registry.restore(rows, list(session.events), 0, dict(HEADER), 0)
    assert restored.snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}


def test_restore_with_mismatched_row_above_base_raises(tmp_path: Path) -> None:
    """restore() above seq 0 with an unusable row raises to re-read from 0."""
    _ctx, registry = _registry()
    session = _golden_session()
    rows = registry.checkpoint(session)
    rows["title"]["ver"] = 99
    with pytest.raises(ValueError, match="re-read from seq 0"):
        registry.restore(rows, list(session.events), 5, dict(HEADER), 0)


def test_cache_never_ahead_of_log_after_put_failure(
    tmp_path: Path, caplog: Any, monkeypatch: Any
) -> None:
    """A crash between log flush and cache write leaves the cache behind."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(tmp_path, registry, ctx)
    events = _golden_events()
    session = _session_with(events[:8])
    _emit(ctx, session, events[:8])
    assert _row_seqs(tmp_path)["title"] == 7
    monkeypatch.setattr(cache, "put", _raising_put)
    title_event = session.append(
        "session/title",
        {"title": "Hello world", "messageSeqs": [1], "source": {"kind": "fallback"}},
    )
    ctx.emit("session/event", session, title_event)
    tail_end = session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    with caplog.at_level(logging.WARNING, logger="harness.session"):
        ctx.emit("session/event", session, tail_end)
    assert 'turn/end write for "s1" failed' in caplog.text
    # The on-disk record is still the older cut.
    assert _row_seqs(tmp_path)["title"] == 7
    monkeypatch.undo()
    snapshot = cache.cold_snapshot(dict(HEADER), 0, list(session.events))
    # No working-directory/change was appended here, so the directory
    # stays at the header cwd.
    expected = {**GOLDEN_VALUES, "workingDirectory": "/home/malcom/work"}
    assert snapshot == {"as_of_seq": 9, "values": expected}
    cache.dispose()


def test_dispose_hazard_row_past_log_end_is_discarded(tmp_path: Path) -> None:
    """A row claiming a seq past the durable log end never seeds a fold.

    The corrected §5.5 step 3: restore_floor gates on ver only, so it
    returns the row's seq; restore itself detects the shrink (row.seq >
    end_seq makes the row unusable, and with base_seq > 0 it raises to
    re-read from seq 0); the refold from seq 0 returns correct values.
    """
    _ctx, registry = _registry()
    session = _golden_session()
    cache = _cache(tmp_path, registry)
    rows = registry.checkpoint(session)
    for row in rows.values():
        row["seq"] = 99
    cache.put("s1", identity_of(dict(HEADER), 0), rows)
    checkpoint = cache.record_for("s1", identity_of(dict(HEADER), 0))["rows"]
    # The floor is ver-gated only: it cannot see the shrink.
    assert registry.restore_floor(checkpoint) == 99
    with pytest.raises(ValueError, match="re-read from seq 0"):
        registry.restore(checkpoint, list(session.events), 99, dict(HEADER), 0)
    # The cold read refolds from seq 0 and returns correct values.
    snapshot = cache.cold_snapshot(dict(HEADER), 0, list(session.events))
    assert snapshot == {"as_of_seq": 9, "values": GOLDEN_VALUES}
    assert _row_seqs(tmp_path)["title"] == 9


def test_on_disk_cache_seq_never_exceeds_log_seq(tmp_path: Path) -> None:
    """After the turn/end write, every cached row is at or behind the log."""
    ctx, registry = _registry()
    cache = _cache_with_listeners(tmp_path, registry, ctx)
    events = _golden_events()
    session = _session_with(events[:8])
    _emit(ctx, session, events[:8])
    last_durable_seq = len(session.events) - 1
    for seq in _row_seqs(tmp_path).values():
        assert seq <= last_durable_seq
    cache.dispose()


# ---------------------------------------------------------------------------
# Service wiring: cache construction, load/read faces, config.
# ---------------------------------------------------------------------------


def _append_golden_live(service: SessionService, session: Session) -> None:
    """Drive Scenario A through the live append path (v4 first-class shapes)."""
    service.log_event(session, "turn/start", {"turn": 1})
    service.log_event(
        session,
        "user/message",
        {
            "content": [{"type": "text", "text": "Hello world"}],
            "source": {"kind": "user"},
        },
    )
    service.log_event(session, "step/start", {"turn": 1, "step": 1})
    step_start_ms = int(session.events[-1].timestamp * 1000)
    service.log_event(
        session,
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "a1",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there"}],
            },
            "stream": [
                {
                    "type": "chunk",
                    "time": step_start_ms,
                    "chunk": {"type": "text-delta", "index": 0, "text": "Hi"},
                }
            ],
            "usage": {"outputTokens": 5},
        },
    )
    service.log_event(
        session,
        "tool/call",
        {"turn": 1, "step": 1, "callId": "c1", "name": "add", "arguments": {}},
    )
    service.log_event(
        session,
        "tool/result",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": "m1",
                "role": "tool",
                "source": {"kind": "tool", "callId": "c1"},
                "toolCallId": "c1",
                "content": [{"type": "text", "text": "3"}],
            },
        },
    )
    service.log_event(session, "step/end", {"turn": 1, "step": 1})
    service.log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    service.log_event(
        session,
        "session/title",
        {"title": "Hello world", "messageSeqs": [1], "source": {"kind": "fallback"}},
    )
    service.log_event(session, "working-directory/change", {"cwd": "/tmp/proj"})


def _assert_live_golden(snapshot: dict[str, Any]) -> None:
    """The live golden log folds to the Scenario A values (wall-clock times)."""
    assert snapshot["as_of_seq"] == 9
    values = snapshot["values"]
    assert values["title"] == "Hello world"
    assert values["turnOutline"] == [
        {"turn": 1, "seq": 0, "prompt": "Hello world", "response": "Hi there"}
    ]
    assert values["workingDirectory"] == "/tmp/proj"
    stats = values["sessionStats"]
    assert stats["turns"] == 1
    assert stats["steps"] == 1
    assert stats["ttft_steps"] == 1
    assert stats["decode_tokens"] == 5
    for key in ("llm_ms", "tool_ms", "ttft_ms", "decode_ms"):
        assert 0 <= stats[key] < 60000


def test_service_memory_only_has_no_cache() -> None:
    """A memory-only service runs without a cache."""
    service = SessionService(Context())
    assert service.projection_cache is None


def test_service_create_writes_create_point_record(tmp_path: Path) -> None:
    """Service creation installs the cache and writes the seed cut."""
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    assert service.projection_cache is not None
    session = service.create("s1")
    path = _record_path(tmp_path)
    assert path.exists()
    record = _read_record(tmp_path)["record"]
    assert record["identity"]["created_at"] == session.header["createdAt"]
    assert record["identity"]["is_seeded"] is False
    assert record["identity"]["inherited_event_count"] == 0
    # The empty log folds to seq -1 on every row.
    assert {row["seq"] for row in record["rows"].values()} == {-1}
    service.projection_cache.dispose()


def test_service_golden_live_write_points(tmp_path: Path) -> None:
    """turn/end writes after the log flush; detach writes the final cut."""
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    session = service.create("s1")
    _append_golden_live(service, session)
    # The turn/end write landed at seq 7; the title/cwd tail stayed dirty.
    assert _row_seqs(tmp_path)["title"] == 7
    assert service.projections.snapshot(session)["as_of_seq"] == 9
    _assert_live_golden(service.projections.snapshot(session))
    service.close("s1")
    assert _row_seqs(tmp_path) == {
        "title": 9,
        "titleInput": 9,
        "sessionStats": 9,
        "turnOutline": 9,
        "workingDirectory": 9,
    }
    # Close drains the tracker's live buffer and flushes the handle, so the
    # whole log is durable now.
    log_path = service.location("s1")
    assert log_path is not None
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "session"
    assert any('"turn/end"' in line for line in lines[1:])
    assert any('"session/title"' in line for line in lines[1:])
    assert service.projection_cache is not None
    service.projection_cache.dispose()


def test_service_turn_end_write_is_at_seq_seven(tmp_path: Path) -> None:
    """The turn/end cache row claims exactly the turn/end cut, never more."""
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    session = service.create("s1")
    service.log_event(session, "turn/start", {"turn": 1})
    service.log_event(
        session,
        "user/message",
        {
            "content": [{"type": "text", "text": "Hello world"}],
            "source": {"kind": "user"},
        },
    )
    service.log_event(session, "step/start", {"turn": 1, "step": 1})
    service.log_event(session, "step/end", {"turn": 1, "step": 1})
    service.log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    assert _row_seqs(tmp_path)["title"] == 4
    assert _row_seqs(tmp_path)["sessionStats"] == 4
    service.close("s1")
    assert service.projection_cache is not None
    service.projection_cache.dispose()


def test_service_load_hydrates_from_cache(tmp_path: Path) -> None:
    """A cold load seeds its cells from the cache and serves live values."""
    first = SessionService(Context(), {"base_storage_dir": tmp_path})
    session = first.create("s1")
    _append_golden_live(first, session)
    first.close("s1")
    assert first.projection_cache is not None
    first.projection_cache.dispose()
    second = SessionService(Context(), {"base_storage_dir": tmp_path})
    loaded = second.load("s1")
    assert loaded is not None
    _assert_live_golden(second.projections.snapshot(loaded))
    cached = second.projections.cached_snapshot(loaded)
    assert cached is not None
    assert cached["as_of_seq"] == 9
    assert second.projection_cache is not None
    second.projection_cache.dispose()


def test_service_read_projections_cold_read(tmp_path: Path) -> None:
    """read_projections cold-reads a stored session and writes the row back."""
    first = SessionService(Context(), {"base_storage_dir": tmp_path})
    session = first.create("s1")
    _append_golden_live(first, session)
    first.close("s1")
    assert first.projection_cache is not None
    first.projection_cache.dispose()
    second = SessionService(Context(), {"base_storage_dir": tmp_path})
    # Remove the record from disk: the cold read must rewrite it.
    _record_path(tmp_path).unlink()
    snapshot = second.read_projections("s1")
    assert snapshot is not None
    _assert_live_golden(snapshot)
    assert _record_path(tmp_path).exists()
    assert _row_seqs(tmp_path)["title"] == 9
    assert second.projection_cache is not None
    second.projection_cache.dispose()


def test_service_read_projections_live_and_missing(tmp_path: Path) -> None:
    """read_projections reads live sessions from the registry, missing as None."""
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    session = service.create("s1")
    _append_golden_live(service, session)
    live = service.read_projections("s1")
    assert live is not None
    _assert_live_golden(live)
    assert service.read_projections("nope") is None
    service.close("s1")
    assert service.read_projections("s1") is not None
    assert service.projection_cache is not None
    service.projection_cache.dispose()


def test_service_read_projections_store_only_session(tmp_path: Path) -> None:
    """read_projections reads a store-live session from the registry."""
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    session = service.store.create("store-only")
    assert session is not None
    snapshot = service.read_projections("store-only")
    assert snapshot is not None
    assert snapshot["as_of_seq"] == -1
    assert snapshot["values"]["title"] is None
    assert service.projection_cache is not None
    service.projection_cache.dispose()


def test_sessions_dir_property(tmp_path: Path) -> None:
    """The sessions directory sits under the cache domain root."""
    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    assert cache.sessions_dir == tmp_path / "session_projcache" / "sessions"


def test_dispose_tolerates_non_callable_disposers(tmp_path: Path) -> None:
    """Dispose skips listeners whose context offers no unregister call."""

    class _StubContext:
        def on(self, *args: Any, **kwargs: Any) -> None:
            return None

    _, registry = _registry()
    cache = _cache(tmp_path, registry)
    cache.install(_StubContext())  # type: ignore[arg-type]
    cache.dispose()
    cache.dispose()


def test_service_read_projections_memory_only(tmp_path: Path) -> None:
    """A memory-only service has no stored session to cold-read."""
    service = SessionService(Context())
    assert service.read_projections("s1") is None


def test_service_cache_config_validation(tmp_path: Path) -> None:
    """Bad cache throttles fail service construction loudly."""
    with pytest.raises(ValueError, match="write_every_events"):
        SessionService(
            Context(),
            {"base_storage_dir": tmp_path, "projection_cache_write_every_events": 0},
        )
    with pytest.raises(ValueError, match="write_interval_ms"):
        SessionService(
            Context(),
            {"base_storage_dir": tmp_path, "projection_cache_write_interval_ms": 0},
        )


def test_service_cache_custom_count_threshold(tmp_path: Path) -> None:
    """The service passes its cache config through to the cache."""
    service = SessionService(
        Context(),
        {
            "base_storage_dir": tmp_path,
            "projection_cache_write_every_events": 3,
            "projection_cache_write_interval_ms": 5000,
        },
    )
    session = service.create("s1")
    service.log_event(session, "turn/start", {"turn": 1})
    service.log_event(
        session,
        "user/message",
        {
            "content": [{"type": "text", "text": "Hello world"}],
            "source": {"kind": "user"},
        },
    )
    assert _row_seqs(tmp_path)["title"] == -1
    service.log_event(session, "step/start", {"turn": 1, "step": 1})
    assert _row_seqs(tmp_path)["title"] == 2
    service.close("s1")
    assert service.projection_cache is not None
    service.projection_cache.dispose()


def test_service_delete_and_recreate_never_seeds_stale_state(
    tmp_path: Path,
) -> None:
    """A deleted-then-recreated id never seeds state from the unrelated log."""
    first = SessionService(Context(), {"base_storage_dir": tmp_path})
    session = first.create("s1")
    _append_golden_live(first, session)
    first.close("s1")
    assert first.delete("s1") is True
    assert first.projection_cache is not None
    first.projection_cache.dispose()
    second = SessionService(Context(), {"base_storage_dir": tmp_path})
    recreated = second.create("s1")
    snapshot = second.projections.snapshot(recreated)
    assert snapshot["values"]["title"] is None
    assert snapshot["values"]["turnOutline"] == []
    second.log_event(recreated, "turn/start", {"turn": 1})
    second.log_event(
        recreated,
        "user/message",
        {
            "content": [{"type": "text", "text": "Fresh start"}],
            "source": {"kind": "user"},
        },
    )
    # The outline prompt comes from the new log only.
    assert second.projections.snapshot(recreated)["values"]["turnOutline"] == [
        {"turn": 1, "seq": 0, "prompt": "Fresh start", "response": ""}
    ]
    assert second.projections.snapshot(recreated)["values"]["title"] is None
    second.close("s1")
    assert second.projection_cache is not None
    second.projection_cache.dispose()
