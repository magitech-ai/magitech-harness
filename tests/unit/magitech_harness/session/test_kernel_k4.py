"""Chunk-level tests for the persistence + projection-cache rows (K4).

The JSONL persistence backend and the projection cache mount as bundle
rows now; SessionService keeps only the store and resolves both lazily
from their slots. These tests pin the chunk's shape: the rows are
Service classes that activate hollow without a root, a storage dir with
no active persistence row fails loud instead of running memory-only,
a swapped bundle still wires, and memory-only boot keeps working.
"""

from pathlib import Path

import pytest
from anima import Context
from anima.jsexpr import dump_yaml, load_yaml

import magitech_harness
from magitech_harness import check_bundle_entries, create_harness
from magitech_harness.session import service as service_module
from magitech_harness.session.events import EventType
from magitech_harness.session.persistence_jsonl import (
    JsonlSessionPersistence,
    JsonlSessionPersistenceService,
)
from magitech_harness.session.projection_cache import (
    ProjectionCacheConfig,
    SessionProjectionCache,
    SessionProjectionCacheService,
)
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService


def _base_rows() -> list[dict]:
    """The default bundle rows as loaded YAML."""
    bundle_path = (
        Path(magitech_harness.__file__).resolve().parent / "profiles" / "base.yaml"
    )
    return load_yaml(bundle_path.read_text(encoding="utf-8"))


def _write_bundle(tmp_path: Path, rows: list[dict]) -> Path:
    """Write rows as a bundle file and return its path."""
    bundle_path = tmp_path / "bundle.yaml"
    with bundle_path.open("w", encoding="utf-8") as handle:
        dump_yaml(rows, handle)
    return bundle_path


def test_service_constructs_neither_persistence_nor_cache() -> None:
    """SessionService no longer constructs the backend, tracker, or cache."""
    source = Path(service_module.__file__).read_text(encoding="utf-8")
    for construction in (
        "JsonlSessionPersistence(",
        "JsonlLiveTracker(",
        "SessionProjectionCache(",
        "ProjectionCacheConfig(",
    ):
        assert construction not in source


def test_backend_owns_tracker_and_root(tmp_path: Path) -> None:
    """The backend exposes its root and owns a tracker it can install."""
    backend = JsonlSessionPersistence(tmp_path)
    assert backend.root == tmp_path
    backend.drain_live()
    backend.drain_live("missing")
    ctx = Context()
    dispose = backend.install_tracker(ctx)
    dispose()
    dispose()


def test_persistence_row_hollow_without_root() -> None:
    """A persistence row with no root activates hollow: no backend."""
    row = JsonlSessionPersistenceService(Context())
    assert row.backend is None
    assert row.active is False
    row.dispose()
    row.dispose()


async def test_persistence_row_with_root_installs_tracker(tmp_path: Path) -> None:
    """A persistence row with a root routes live events onto the log."""
    ctx = Context()
    row = JsonlSessionPersistenceService(ctx, {"root": tmp_path})
    assert row.active is True
    assert row.backend is not None
    assert row.backend.root == tmp_path
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    session = service.create("k4-row")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "first"})
    await service.flush(session)
    location = service.location("k4-row")
    assert location is not None
    assert "first" in location.read_text(encoding="utf-8")
    # Row unload uninstalls the tracker: later events never reach the log.
    row.dispose()
    service.log_event(session, EventType.USER_MESSAGE, {"text": "second"})
    await service.flush(session)
    assert "second" not in location.read_text(encoding="utf-8")


def test_cache_row_hollow_without_root() -> None:
    """A cache row with no root activates hollow: no cache."""
    row = SessionProjectionCacheService(Context())
    assert row.cache is None
    assert row.active is False
    row.dispose()
    row.dispose()


def test_cache_row_requires_sessions_and_registry(tmp_path: Path) -> None:
    """The cache row fails loud when either injected service is missing."""
    with pytest.raises(
        RuntimeError, match="session-projection-cache requires the sessions service"
    ):
        SessionProjectionCacheService(Context(), {"root": tmp_path})
    ctx = Context()
    SessionService(ctx)
    with pytest.raises(
        RuntimeError,
        match="session-projection-cache requires the sessionProjections service",
    ):
        SessionProjectionCacheService(ctx, {"root": tmp_path})


async def test_cache_row_with_root_installs_cache(tmp_path: Path) -> None:
    """A cache row with a root checkpoints live sessions through the service."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    JsonlSessionPersistenceService(ctx, {"root": tmp_path})
    service = SessionService(ctx, {"base_storage_dir": tmp_path})
    row = SessionProjectionCacheService(ctx, {"root": tmp_path})
    assert row.active is True
    assert row.cache is not None
    assert service.projection_cache is row.cache
    service.create("k4-cache")
    record = tmp_path / "session_projcache" / "sessions" / "k4-cache.json"
    assert record.exists()
    row.dispose()
    assert row.cache is None
    assert row.active is False
    assert service.projection_cache is None


async def test_session_op_fails_loud_without_persistence_row(
    tmp_path: Path,
) -> None:
    """A storage dir with no persistence row fails loud, never memory-only."""
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    with pytest.raises(RuntimeError, match="sessionPersistence"):
        service.create("k4-loud")
    with pytest.raises(RuntimeError, match="sessionPersistence"):
        service.list_sessions()
    with pytest.raises(RuntimeError, match="sessionPersistence"):
        service.load("k4-loud")
    with pytest.raises(RuntimeError, match="sessionPersistence"):
        service.read("k4-loud")
    with pytest.raises(RuntimeError, match="sessionPersistence"):
        await service.flush()
    assert list(tmp_path.iterdir()) == []
    # Memory-only stays memory-only: the same ops work with no root.
    memory_only = SessionService(Context())
    session = memory_only.create("k4-memory")
    assert memory_only.list_sessions() == ["k4-memory"]
    assert memory_only.load("missing") is None
    assert memory_only.read("missing") is None
    await memory_only.flush()
    assert memory_only.close(session.session_id) is True


async def test_flush_log_for_cache_fails_loud_without_persistence(
    tmp_path: Path,
) -> None:
    """The cache's log flush fails loud on a memory-only service."""
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    service = SessionService(ctx)
    session = service.create("k4-flush-log")
    cache = SessionProjectionCache(
        registry,
        tmp_path,
        ProjectionCacheConfig(write_every_events=200, write_interval_ms=5000),
        flush_log=service.flush_log_for_cache,
    )
    with pytest.raises(RuntimeError, match="sessionPersistence"):
        cache.write(session)


async def test_memory_only_flush_all_is_a_noop() -> None:
    """A memory-only service-wide flush drains nothing and raises nothing."""
    service = SessionService(Context())
    session = service.create("k4-noop")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "hi"})
    await service.flush()
    assert service.get("k4-noop") is session


async def test_swapped_cache_row_still_wires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache row does not depend on bundle order for its injections.

    With the cache row listed before the sessions row, the cache still
    waits for sessions and the registry, still installs, and still
    answers projection reads live and cold.
    """
    rows = _base_rows()
    cache = next(
        i for i, row in enumerate(rows) if row["id"] == "session-projection-cache"
    )
    sess = next(i for i, row in enumerate(rows) if row["id"] == "sessions")
    rows[cache], rows[sess] = rows[sess], rows[cache]
    monkeypatch.setattr(
        magitech_harness, "_BASE_BUNDLE_PATH", _write_bundle(tmp_path, rows)
    )
    ctx = create_harness(tmp_path / "sessions")
    await ctx.flush()
    check_bundle_entries(ctx)
    service = ctx.sessions
    assert service.projection_cache is not None
    session = service.create("k4-swapped")
    service.log_event(session, "turn/start", {"turn": 1})
    assert service.read_projections("k4-swapped") is not None
    assert service.close("k4-swapped") is True
    assert service.read_projections("k4-swapped") is not None


async def test_bundle_without_persistence_row_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle with a storage dir but no persistence row fails loud.

    Every row still activates, so the bundle audit passes; the first
    session op then refuses instead of silently running memory-only.
    """
    rows = [row for row in _base_rows() if row["id"] != "session-persistence-jsonl"]
    monkeypatch.setattr(
        magitech_harness, "_BASE_BUNDLE_PATH", _write_bundle(tmp_path, rows)
    )
    ctx = create_harness(tmp_path / "sessions")
    await ctx.flush()
    check_bundle_entries(ctx)
    with pytest.raises(RuntimeError, match="sessionPersistence"):
        ctx.sessions.create("k4-no-row")


async def test_hollow_persistence_row_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistence row with no root is hollow, and session ops fail loud."""
    rows = _base_rows()
    persistence = next(row for row in rows if row["id"] == "session-persistence-jsonl")
    persistence["config"] = {}
    monkeypatch.setattr(
        magitech_harness, "_BASE_BUNDLE_PATH", _write_bundle(tmp_path, rows)
    )
    ctx = create_harness(tmp_path / "sessions")
    await ctx.flush()
    check_bundle_entries(ctx)
    assert ctx.sessions.base_storage_dir is not None
    assert ctx.get("sessionPersistence", None, strict=False).backend is None
    with pytest.raises(RuntimeError, match="sessionPersistence"):
        ctx.sessions.create("k4-hollow")


async def test_memory_only_boot_rows_are_hollow() -> None:
    """Memory-only boot activates both rows hollow and sessions still work."""
    ctx = create_harness()
    await ctx.flush()
    check_bundle_entries(ctx)
    persistence = ctx.get("sessionPersistence", None, strict=False)
    assert persistence is not None
    assert persistence.backend is None
    assert persistence.active is False
    cache = ctx.get("sessionProjectionCache", None, strict=False)
    assert cache is not None
    assert cache.cache is None
    assert cache.active is False
    assert ctx.sessions.projection_cache is None
    service = ctx.sessions
    session = service.create("k4-memory-boot")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "hi"})
    await service.flush()
    assert service.close("k4-memory-boot") is True
    assert service.location("k4-memory-boot") is None


async def test_bundle_slots_with_storage(tmp_path: Path) -> None:
    """With a storage dir the bundle rows own a backend and a cache."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    check_bundle_entries(ctx)
    persistence = ctx.get("sessionPersistence", None, strict=False)
    assert persistence is not None
    assert persistence.backend is not None
    assert persistence.backend.root == tmp_path
    cache = ctx.get("sessionProjectionCache", None, strict=False)
    assert cache is not None
    assert cache.cache is not None
    assert cache.cache.sessions_dir.is_relative_to(tmp_path)
    service = ctx.sessions
    assert service.projection_cache is cache.cache
    session = service.create("k4-slots")
    service.log_event(session, EventType.USER_MESSAGE, {"text": "hi"})
    await service.flush()
    location = service.location("k4-slots")
    assert location is not None
    assert location.exists()
    assert service.read_projections("k4-slots") is not None
