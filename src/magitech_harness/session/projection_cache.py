"""Persisted projection cache: durable per-session checkpoint records.

One record per session lives under ``<base_storage_dir>/session_projcache/
sessions/<encoded-id>.json``. A record binds its rows to the log identity
that produced them, so a record folded from an unrelated log can never seed
a fold. Records are disposable derived data: a corrupt record is moved
aside at open and the session boots uncached; a failed write only costs a
longer tail replay on the next cold read.

Ordering: every durable write flushes the session log first, so a crash
may leave the cache behind the log, never ahead of it. The restore
backstop (a row whose ``seq`` is past the durable log end is discarded
and the session refolds from seq 0) covers the dispose window, where the
store entry is already gone when the final write lands.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeGuard
from uuid import uuid4

from anima import Context, Service

from magitech_harness.session.events import (
    SessionEvent,
    is_lossless_json,
    snapshot_json_value,
)
from magitech_harness.session.persistence_jsonl import (
    _ensure_session_dir,
    _fsync_dir,
    encode_segment,
)
from magitech_harness.session.projection_registry import (
    CheckpointRow,
    ProjectionSnapshot,
    SessionProjectionRegistry,
)

if TYPE_CHECKING:
    from magitech_harness.session.store import Session

logger = logging.getLogger("harness.session")

#: Domain name: one table (``sessions``) keyed by session id.
_DOMAIN_NAME = "session_projcache"
#: Current domain version, stamped into every written envelope.
_DOMAIN_VERSION = 7
#: Envelope stamps a reader accepts: the current version plus compatible ones.
_ACCEPTED_VERSIONS = frozenset({7, 3, 4, 5, 6})
#: Table directory below the domain root.
_TABLE_DIR = "sessions"


@dataclass(frozen=True)
class ProjectionCacheConfig:
    """Throttle triggers for the write-behind. Both required, no defaults.

    The service supplies them from its own config, defaulting to the
    shipped 200 events / 5000 ms; the cache class itself has no defaults.
    """

    write_every_events: int
    write_interval_ms: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.write_every_events, bool)
            or not isinstance(self.write_every_events, int)
            or self.write_every_events < 1
        ):
            raise ValueError(
                "projection cache write_every_events must be an integer >= 1, "
                f"got {self.write_every_events!r}"
            )
        if (
            isinstance(self.write_interval_ms, bool)
            or not isinstance(self.write_interval_ms, int)
            or self.write_interval_ms < 1
        ):
            raise ValueError(
                "projection cache write_interval_ms must be an integer >= 1, "
                f"got {self.write_interval_ms!r}"
            )


def lifecycle_identity_of(header: dict[str, Any]) -> dict[str, Any]:
    """The identity fields a header alone can witness (no inherited cut)."""
    identity: dict[str, Any] = {
        "format_version": header["version"],
        "created_at": header["createdAt"],
        "is_seeded": header["isSeeded"],
    }
    cwd = header.get("cwd")
    if cwd is not None:
        identity["cwd"] = cwd
    return identity


def identity_of(header: dict[str, Any], inherited_event_count: int) -> dict[str, Any]:
    """The complete fold identity: lifecycle fields plus the inherited cut."""
    if (
        isinstance(inherited_event_count, bool)
        or not isinstance(inherited_event_count, int)
        or inherited_event_count < 0
    ):
        raise ValueError(
            "projection-cache identity inherited event count must be an "
            f"integer >= 0, got {inherited_event_count!r}"
        )
    if not header.get("isSeeded") and inherited_event_count != 0:
        raise ValueError(
            "unseeded projection-cache identity inherited event count must be 0"
        )
    identity = lifecycle_identity_of(header)
    identity["inherited_event_count"] = inherited_event_count
    return identity


def _lifecycle_fields_match(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Whether the format-independent lifecycle fields agree.

    An absent ``is_seeded`` reads as unseeded; an absent ``cwd`` matches
    only an absent one (both read as ``None`` here).
    """
    stored_seeded = stored.get("is_seeded")
    return (
        stored.get("created_at") == expected.get("created_at")
        and stored.get("cwd") == expected.get("cwd")
        and (stored_seeded if stored_seeded is not None else False)
        == expected.get("is_seeded")
    )


def current_lifecycle_matches(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Whether a stored record was folded from the caller's lifecycle.

    An absent ``format_version`` never matches: no current log can prove
    that record's fold semantics. This is the whole identity a
    header-only reader can check, and the whole identity a view needs.
    """
    return stored.get("format_version") == expected.get(
        "format_version"
    ) and _lifecycle_fields_match(stored, expected)


def identity_matches(stored: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Whether a stored record may seed the caller's fold (lifecycle + cut).

    A record folded under another cut encodes that cut in unit states and
    would carry it into the continued fold. An absent cut reads as the
    unseeded lineage: exact for an unseeded caller, while a seeded caller
    fails the match.
    """
    stored_cut = stored.get("inherited_event_count")
    return current_lifecycle_matches(stored, expected) and (
        stored_cut if stored_cut is not None else 0
    ) == expected.get("inherited_event_count")


def predecessor_identity_matches(
    stored: dict[str, Any], expected: dict[str, Any]
) -> bool:
    """Whether a stored record is a predecessor of the caller's lifecycle."""
    stored_version = stored.get("format_version")
    predecessor = stored_version is None or (
        isinstance(stored_version, int)
        and not isinstance(stored_version, bool)
        and stored_version < expected["format_version"]
    )
    return predecessor and _lifecycle_fields_match(stored, expected)


def _is_int(value: Any) -> TypeGuard[int]:
    """Whether ``value`` is a genuine integer (a bool is not a number here)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _identity_error(identity: Any) -> str | None:
    """The first schema failure of a stored identity, or ``None`` when valid."""
    if not isinstance(identity, dict):
        return "identity is not an object"
    created_at = identity.get("created_at")
    if not _is_int(created_at) or created_at < 0:
        return "identity.created_at must be an integer >= 0"
    format_version = identity.get("format_version")
    if "format_version" in identity and (
        not _is_int(format_version) or format_version < 0
    ):
        return "identity.format_version must be an integer >= 0 when present"
    cwd = identity.get("cwd")
    if "cwd" in identity and not isinstance(cwd, str):
        return "identity.cwd must be a string when present"
    is_seeded = identity.get("is_seeded")
    if "is_seeded" in identity and not isinstance(is_seeded, bool):
        return "identity.is_seeded must be a boolean when present"
    inherited = identity.get("inherited_event_count")
    if "inherited_event_count" in identity and (
        not _is_int(inherited) or inherited < 0
    ):
        return "identity.inherited_event_count must be an integer >= 0 when present"
    return None


def _row_error(row: Any) -> str | None:
    """The first schema failure of one stored checkpoint row, or ``None``."""
    if not isinstance(row, dict):
        return "row is not an object"
    ver = row.get("ver")
    if not _is_int(ver) or ver < 0:
        return "row.ver must be an integer >= 0"
    seq = row.get("seq")
    if not _is_int(seq) or seq < -1:
        return "row.seq must be an integer >= -1"
    if "val" not in row:
        return "row.val is missing"
    if not is_lossless_json(row["val"]):
        return "row.val must be losslessly JSON-serializable"
    return None


def _record_error(record: Any) -> str | None:
    """The first schema failure of a stored record, or ``None`` when valid."""
    if not isinstance(record, dict):
        return "record is not an object"
    error = _identity_error(record.get("identity"))
    if error is not None:
        return error
    rows = record.get("rows")
    if not isinstance(rows, dict):
        return "rows is not an object"
    for key, row in rows.items():
        error = _row_error(row)
        if error is not None:
            return f'row "{key}": {error}'
    return None


def _read_document(path: Path) -> tuple[str, Any]:
    """Classify one stored document.

    Returns ``("foreign", None)`` for a document that is malformed or
    stamped with an unaccepted version (reads as absent, silently),
    ``("invalid", reason)`` for an accepted stamp whose record fails the
    schema, and ``("valid", record)`` otherwise.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ("foreign", None)
    try:
        document = json.loads(text)
    except ValueError:
        return ("foreign", None)
    if not isinstance(document, dict):
        return ("foreign", None)
    stamp = document.get("version")
    if not _is_int(stamp) or stamp not in _ACCEPTED_VERSIONS:
        return ("foreign", None)
    if "record" not in document:
        return ("invalid", "record is missing")
    record = document["record"]
    error = _record_error(record)
    if error is not None:
        return ("invalid", error)
    return ("valid", record)


def _backup_stamp() -> str:
    """Local-time ``YYYYMMDDHHmm`` suffix for a backed-up document."""
    now = time.localtime()
    return (
        f"{now.tm_year:04d}{now.tm_mon:02d}{now.tm_mday:02d}"
        f"{now.tm_hour:02d}{now.tm_min:02d}"
    )


def _write_atomic(path: Path, text: str) -> None:
    """Durably replace one document: temp file, fsync, atomic rename.

    The temp file is opened ``O_CREAT | O_EXCL`` with mode ``0o600`` in
    the same directory; the rename is atomic, so the record is old or
    new, never torn; the containing directory is fsynced after the
    rename. Both directories are created with mode ``0o700``. On any
    failure the temp file is removed and the error propagates.
    """
    _ensure_session_dir(path.parent.parent)
    _ensure_session_dir(path.parent)
    tmp = path.parent / f".{uuid4().hex}.tmp"
    created = False
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        created = True
        try:
            data = text.encode("utf-8")
            while data:
                written = os.write(fd, data)
                data = data[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        created = False
        _fsync_dir(path.parent)
    except Exception:
        if created:
            os.unlink(tmp)
        raise


@dataclass
class _DirtyState:
    """Per-session write-behind bookkeeping (live sessions only; dropped at retire)."""

    pending: int = 0
    timer: threading.Timer | None = None


class SessionProjectionCache:
    """The persisted projection cache for one persistence root.

    Opens the ``session_projcache`` domain at construction: every stored
    document is read once, schema-invalid records are moved aside and
    skipped, and the surviving records are held in memory. Every read
    serves that in-memory table; no read touches disk. Writes are
    fail-soft through :meth:`flush_soft`; :meth:`write` itself is not.
    """

    def __init__(
        self,
        registry: SessionProjectionRegistry,
        base_storage_dir: Path | str,
        config: ProjectionCacheConfig,
        *,
        flush_log: Callable[[Session], None] | None = None,
    ) -> None:
        self._registry = registry
        self._config = config
        self._flush_log = flush_log
        self._base_storage_dir = Path(base_storage_dir)
        self._sessions_dir = self._base_storage_dir / _DOMAIN_NAME / _TABLE_DIR
        self._records: dict[str, dict[str, Any]] = {}
        self._dirty: dict[str, _DirtyState] = {}
        self._disposers: list[Any] = []
        self._installed = False
        self._load_all()

    @property
    def sessions_dir(self) -> Path:
        """The directory holding the per-session record documents."""
        return self._sessions_dir

    def _record_path(self, session_id: str) -> Path:
        """The document path for one session id."""
        return self._sessions_dir / f"{encode_segment(session_id)}.json"

    def _load_all(self) -> None:
        """Scan every stored document once; back up invalid records and skip them.

        A record that fails the schema is disposable derived data, so it
        must never cost the boot: its document is moved aside, the
        concrete failure is logged, and the session is served as
        uncached. A backup that cannot be moved fails loud instead.
        """
        sessions_dir = self._sessions_dir
        if not sessions_dir.is_dir():
            return
        for path in sorted(sessions_dir.iterdir()):
            if not path.is_file() or not path.name.endswith(".json"):
                continue
            key = path.name[: -len(".json")]
            kind, value = _read_document(path)
            if kind == "foreign":
                continue
            if kind == "invalid":
                moved = self._backup_record(path)
                logger.error(
                    "domain '%s': stored record '%s' in table '%s' failed "
                    "schema validation; moved to '%s' and treated as "
                    "absent. Cause: %s",
                    _DOMAIN_NAME,
                    key,
                    _TABLE_DIR,
                    moved,
                    value,
                )
                continue
            self._records[key] = value

    def _backup_record(self, path: Path) -> str:
        """Move one document aside; the moved file no longer ends in ``.json``.

        A same-minute backup of the same key overwrites the previous
        backup; the newer bytes are the ones worth keeping.
        """
        moved = f"{path}.bak.{_backup_stamp()}"
        os.replace(path, moved)
        return moved

    def install(self, ctx: Any) -> Callable[[], None]:
        """Register the write-behind listeners; return their disposer.

        Install order matters for the dispose path: the service installs
        the live tracker first, so its final drain precedes the cache's
        detach write.
        """
        if self._installed:
            raise ValueError("projection cache is already installed")
        self._disposers = [
            ctx.on("session/event", self._on_event),
            ctx.on("session/created", self._on_created),
            ctx.on("session/disposed", self._on_disposed),
        ]
        self._installed = True
        return self.dispose

    def dispose(self) -> None:
        """Unregister listeners, cancel pending timers, drop dirty state.

        Idempotent; a fresh install after disposal is allowed.
        """
        if not self._installed:
            return
        self._installed = False
        for disposer in self._disposers:
            if callable(disposer):
                disposer()
        self._disposers = []
        for dirty in self._dirty.values():
            if dirty.timer is not None:
                dirty.timer.cancel()
        self._dirty.clear()

    def record_for(
        self, session_id: str, expected: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The stored record for one session, accepted only on identity match.

        A session id names a slot, not a lifecycle: a recreated id, or a
        persistence root swapped under a surviving cache, must not let an
        old record seed state folded from an unrelated log. A mismatch is
        treated as absent; the record is not deleted, backed up, or
        repaired.
        """
        record = self._records.get(encode_segment(session_id))
        if record is None:
            return None
        return record if identity_matches(record["identity"], expected) else None

    def cached_snapshot(
        self,
        meta: dict[str, Any],
        keys: list[str] | set[str] | frozenset[str] | None = None,
    ) -> ProjectionSnapshot | None:
        """Zero-I/O listing read: values viewed straight from the stored rows.

        The header is the only identity witness a listing holds, so this
        face matches the lifecycle identity and not the inherited cut: a
        viewed value never seeds a fold. ``as_of_seq`` is the lowest
        watermark among the served rows; ``None`` when no usable row
        exists for this lifecycle.
        """
        expected = lifecycle_identity_of(meta)
        record = self._records.get(encode_segment(str(meta["id"])))
        if record is None or not current_lifecycle_matches(
            record["identity"], expected
        ):
            return None
        return self._view_record(record, keys)

    def cached_predecessor_title(
        self, meta: dict[str, Any]
    ) -> ProjectionSnapshot | None:
        """Read only a predecessor record's title as a zero-I/O listing hint.

        Returns a title-only block at the stored title row's watermark,
        or ``None`` when the record is current, newer, unrelated, missing,
        or incompatible with the title unit. No other predecessor
        projection is exposed.
        """
        expected = lifecycle_identity_of(meta)
        record = self._records.get(encode_segment(str(meta["id"])))
        if record is None or not predecessor_identity_matches(
            record["identity"], expected
        ):
            return None
        return self._view_record(record, ["title"])

    def _view_record(
        self,
        record: dict[str, Any],
        keys: list[str] | set[str] | frozenset[str] | None = None,
    ) -> ProjectionSnapshot | None:
        """View selected wire rows as one block bound to the lowest watermark."""
        values = self._registry.view_checkpoint(record["rows"], keys)
        as_of_seq: int | None = None
        for key, row in record["rows"].items():
            if key not in values:
                continue
            seq = row["seq"]
            if as_of_seq is None or seq < as_of_seq:
                as_of_seq = seq
        if as_of_seq is None:
            return None
        return ProjectionSnapshot(as_of_seq=as_of_seq, values=values)

    def hydrate_prepared(
        self, session: Session, events: list[SessionEvent]
    ) -> ProjectionSnapshot:
        """Seed an already-prepared session's cells from identity-checked rows.

        Falls back to a full-log hydrate on any error: cached rows are
        disposable derived data, and a stale schema must never make a
        valid session unreadable. No checkpoint is written here.
        """
        record = self.record_for(
            session.session_id,
            identity_of(session.header, session.inherited_event_count),
        )
        if record is None:
            return self._registry.hydrate(session, {}, events, 0)
        try:
            return self._registry.hydrate(session, record["rows"], events, 0)
        except Exception:  # noqa: BLE001 - retry from the exact log, never fail the boot
            return self._registry.hydrate(session, {}, events, 0)

    def cold_snapshot(
        self,
        meta: dict[str, Any],
        inherited_event_count: int,
        events: list[SessionEvent],
    ) -> ProjectionSnapshot:
        """Cold-read one session's projections from its complete log.

        Each unit is seeded from the identity-checked cached rows, and the
        refreshed checkpoint is written back (fail-soft, fire-and-forget),
        so the first cold read creates the cache row and later ones seed
        from it. The caller supplies the complete log in seq order; this
        service never consults the persistence layer.
        """
        identity = identity_of(meta, inherited_event_count)
        record = self.record_for(str(meta["id"]), identity)
        rows = record["rows"] if record is not None else {}
        restored = self._registry.restore(rows, events, 0, meta, inherited_event_count)
        try:
            self.put(str(meta["id"]), identity, restored.checkpoint)
        except Exception as error:  # noqa: BLE001 - a failed write-back only costs a longer tail replay
            logger.warning(
                'session projection cache: cold-read write-back for "%s" failed '
                "(cache stays stale): %s",
                meta["id"],
                error,
            )
        return restored.snapshot

    def write(self, session: Session) -> None:
        """Durably checkpoint one live session now. Not fail-soft.

        The checkpoint cut is snapshotted first, then the log is flushed,
        then the record lands: a crash can leave the cache behind the log
        (longer tail replay) but never ahead of it (phantom values folded
        from events no stored log contains). The log flush is skipped when
        no write handle exists yet (the create-point write runs before
        the persistence handle exists) or when the session is no longer
        live (the detach path; the store entry is already gone).
        """
        rows = self._registry.checkpoint(session)
        self.mark_clean(session)
        if self._flush_log is not None:
            self._flush_log(session)
        self.put(
            session.session_id,
            identity_of(session.header, session.inherited_event_count),
            rows,
        )

    def flush_soft(self, session: Session, trigger: str) -> None:
        """One fail-soft durable checkpoint.

        Every caller has work by construction: the throttle triggers only
        fire dirty (``mark_clean`` clears the timer with the counter) and
        the mandatory points write unconditionally.
        """
        try:
            self.write(session)
        except Exception as error:  # noqa: BLE001 - a lost write costs a longer tail replay
            logger.warning(
                'session projection cache: %s write for "%s" failed (cache stays '
                "stale): %s",
                trigger,
                session.session_id,
                error,
            )

    def mark_clean(self, session: Session) -> None:
        """Reset one session's dirty bookkeeping (its checkpoint is being written)."""
        state = self._dirty.get(session.session_id)
        if state is None:
            return
        state.pending = 0
        if state.timer is not None:
            state.timer.cancel()
            state.timer = None

    def put(
        self,
        session_id: str,
        identity: dict[str, Any],
        rows: dict[str, CheckpointRow],
    ) -> None:
        """Replace one session's stored record with its log identity and rows.

        The checkpoint must be losslessly JSON-serializable and the
        assembled record must pass the record schema; the document is
        then replaced atomically, and only then does the in-memory table
        change (durability first, then memory).
        """
        detached = snapshot_json_value({"identity": identity, "rows": rows})
        if detached is None:
            raise TypeError(
                "projection checkpoint is not losslessly JSON-serializable "
                "(a unit state violates the plain-JSON contract)"
            )
        error = _record_error(detached)
        if error is not None:
            raise ValueError(
                f'session projection cache: record for "{session_id}" failed '
                f"schema validation: {error}"
            )
        text = json.dumps({"version": _DOMAIN_VERSION, "record": detached}, indent=2)
        text += "\n"
        target = self._record_path(session_id)
        _write_atomic(target, text)
        self._records[encode_segment(session_id)] = detached

    def _on_event(self, session: Session, event: SessionEvent) -> None:
        """Count one committed event; ``turn/end`` is a mandatory write point."""
        try:
            if event.type == "turn/end":
                self.flush_soft(session, "turn/end")
                return
            state = self._dirty.setdefault(session.session_id, _DirtyState())
            state.pending += 1
            if state.pending >= self._config.write_every_events:
                self.flush_soft(session, "count threshold")
                return
            if state.timer is None:
                state.timer = threading.Timer(
                    self._config.write_interval_ms / 1000,
                    lambda: self.flush_soft(session, "interval"),
                )
                state.timer.daemon = True
                state.timer.start()
        except Exception as error:  # noqa: BLE001 - listeners never raise out of the bus
            logger.warning(
                'session projection cache: event listener for "%s" failed '
                "(cache stays stale): %s",
                session.session_id,
                error,
            )

    def _on_created(self, session: Session) -> None:
        """Creation is the first mandatory write point."""
        try:
            self.flush_soft(session, "create")
        except Exception as error:  # noqa: BLE001 - listeners never raise out of the bus
            logger.warning(
                'session projection cache: create listener for "%s" failed '
                "(cache stays stale): %s",
                session.session_id,
                error,
            )

    def _on_disposed(self, session: Session) -> None:
        """Detach is the final mandatory write point; then drop the dirty entry."""
        try:
            self.flush_soft(session, "detach")
        except Exception as error:  # noqa: BLE001 - listeners never raise out of the bus
            logger.warning(
                'session projection cache: detach listener for "%s" failed '
                "(cache stays stale): %s",
                session.session_id,
                error,
            )
        finally:
            self.mark_clean(session)
            self._dirty.pop(session.session_id, None)


class SessionProjectionCacheService(Service):
    """Row service owning the persisted projection cache.

    A row service providing ``sessionProjectionCache``. With a storage
    root configured it constructs the cache, installs its write-behind
    listeners, and wires the sessions service's log flush ahead of
    every cache write; with no root it activates hollow — the slot
    exists but carries no cache. Row unload uninstalls the listeners
    and cancels the write-behind timers.
    """

    inject: ClassVar[list[str]] = ["sessions", "sessionProjections"]

    def __init__(
        self,
        ctx: Context,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Build the service; the cache exists only with a root configured.

        ``config`` carries ``root``, ``write_every_events``, and
        ``write_interval_ms``; a falsy root activates the service
        hollow. Raises ``RuntimeError`` when the context carries no
        sessions service or registry.
        """
        super().__init__(ctx, "sessionProjectionCache")
        cfg = config or {}
        root = cfg.get("root") or None
        self._cache: SessionProjectionCache | None = None
        if root is not None:
            sessions = ctx.get("sessions", None, strict=False)
            if sessions is None:
                raise RuntimeError(
                    "session-projection-cache requires the sessions service"
                )
            registry = ctx.get("sessionProjections", None, strict=False)
            if registry is None:
                raise RuntimeError(
                    "session-projection-cache requires the sessionProjections service"
                )
            cache = SessionProjectionCache(
                registry,
                Path(root),
                ProjectionCacheConfig(
                    write_every_events=cfg.get("write_every_events", 200),
                    write_interval_ms=cfg.get("write_interval_ms", 5000),
                ),
                flush_log=sessions.flush_log_for_cache,
            )
            cache.install(ctx)
            self._cache = cache
        # Row unload runs this: listener removal alone would not stop
        # the write-behind timers.
        ctx.effect(lambda: self.dispose)

    @property
    def cache(self) -> SessionProjectionCache | None:
        """The owned cache, or ``None`` when the row is hollow."""
        return self._cache

    @property
    def active(self) -> bool:
        """Whether this row owns a live cache."""
        return self._cache is not None

    def dispose(self) -> None:
        """Uninstall the cache listeners and cancel its timers. Idempotent."""
        if self._cache is not None:
            cache = self._cache
            self._cache = None
            cache.dispose()
