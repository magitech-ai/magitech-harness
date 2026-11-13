"""Session storage and deterministic model history projection (derive_messages)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from magitech_harness.session.events import EventType, SessionEvent
from magitech_harness.session.format import (
    CURRENT_FORMAT_VERSION,
    SESSION_FORMAT_CATALOG,
    SessionFormatError,
    SessionFormatUnsupportedError,
    SessionFormatUnsupportedMigrationError,
    SessionPersistenceCorruptionError,
    create_session_format_catalog_with_children,
    decode_envelope_row,
    encode_envelope_event,
    is_json_object,
    require_count,
    require_safe_integer,
    session_format_version_refusal,
)


def _normalize_adjacent_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merges adjacent messages with identical roles to satisfy strict model provider APIs."""
    if not messages:
        return []

    normalized: list[dict[str, Any]] = []
    for msg in messages:
        if not normalized:
            normalized.append(dict(msg))
            continue

        prev = normalized[-1]
        if (
            prev.get("role") == msg.get("role")
            and msg.get("role") in ("user", "system", "assistant")
            and "tool_calls" not in prev
            and "tool_calls" not in msg
        ):
            c1 = prev.get("content")
            c2 = msg.get("content")
            s1 = str(c1) if c1 is not None else ""
            s2 = str(c2) if c2 is not None else ""
            if s1 and s2:
                prev["content"] = f"{s1}\n\n{s2}"
            else:
                prev["content"] = s1 or s2
        else:
            normalized.append(dict(msg))

    return normalized


def derive_messages(events: list[SessionEvent]) -> list[dict[str, Any]]:
    """Deterministically reconstruct model-visible conversation history from session log.

    Adheres to: Model-visible ⟺ Logged.
    """
    latest_compaction: SessionEvent | None = None
    latest_compaction_idx = -1
    for i, ev in enumerate(events):
        if ev.type == EventType.COMPACTION:
            latest_compaction = ev
            latest_compaction_idx = i

    messages: list[dict[str, Any]] = []

    if latest_compaction is not None:
        summary = latest_compaction.payload.get("summary", "")
        messages.append(
            {
                "role": "user",
                "content": f"[Conversation summary up to checkpoint]:\n{summary}",
            }
        )
        compacted_count = latest_compaction.payload.get("compacted_count")
        if compacted_count is not None:
            slice_events = [
                ev for ev in events[compacted_count:] if ev.type != EventType.COMPACTION
            ]
        else:
            slice_events = events[latest_compaction_idx + 1 :]
    else:
        slice_events = events

    for ev in slice_events:
        if ev.type == EventType.USER_MESSAGE:
            messages.append(
                {
                    "role": "user",
                    "content": ev.payload.get("text", ""),
                }
            )
        elif ev.type == EventType.ASSISTANT_MESSAGE:
            msg: dict[str, Any] = {
                "role": "assistant",
                "content": ev.payload.get("text", "") or None,
            }
            tool_calls = ev.payload.get("tool_calls")
            if tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["call_id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"])
                            if isinstance(tc["arguments"], dict)
                            else str(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(msg)
        elif ev.type == EventType.TOOL_RESULT:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": ev.payload.get("call_id"),
                    "name": ev.payload.get("name"),
                    "content": json.dumps(ev.payload.get("content"))
                    if not isinstance(ev.payload.get("content"), str)
                    else ev.payload.get("content"),
                }
            )

    return _normalize_adjacent_messages(messages)


def _lock_exclusive(f: TextIO) -> None:
    """Take an exclusive advisory lock on the open session file.

    POSIX uses flock; Windows uses msvcrt byte-range locking on the first
    byte (the single-byte scheme file-locking libraries use). Both are
    blocking and exclusive, so concurrent appenders serialize. The lock
    region is pinned to offset 0 because msvcrt locks from the current
    file position, which moves after a write.
    """
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_EX)


def _unlock(f: TextIO) -> None:
    """Release the advisory lock taken by _lock_exclusive."""
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


class Session:
    """A durable, append-only conversation session."""

    def __init__(self, session_id: str, storage_path: Path | None = None) -> None:
        self.session_id = session_id
        self.storage_path = storage_path
        self.events: list[SessionEvent] = []
        # Retained v4 logical header (set on load, built on first append).
        self._header: dict[str, Any] | None = None
        # True when the on-disk log (if any) is current-version v4. A
        # session loaded from a migrated log refuses append until the log
        # is republished as current-version v4.
        self._on_disk_current: bool = True
        if (
            storage_path is not None
            and storage_path.exists()
            and storage_path.stat().st_size > 0
        ):
            # Classify the log this session would append to. A historical
            # log refuses append from construction; an unsupported or
            # malformed header fails loud here, as it does on load.
            header, self._on_disk_current = _on_disk_header(storage_path)
            if header is not None:
                self._header = header

    def append(self, event: SessionEvent) -> None:
        """Append event to in-memory log and persist to JSONL if configured."""
        needs_header = self._header is None
        if (
            self.storage_path is not None
            and self.storage_path.exists()
            and self.storage_path.stat().st_size > 0
        ):
            # The log may predate this session (constructed over an existing
            # file) or have been replaced on disk after load: re-classify
            # its header before any mutation, so append still refuses a
            # historical log instead of bricking it.
            on_disk_header, on_disk_current = _on_disk_header(self.storage_path)
            self._on_disk_current = on_disk_current
            needs_header = self._header is None and on_disk_header is None
            if on_disk_header is not None and self._header is None:
                self._header = on_disk_header
        if not self._on_disk_current:
            raise SessionFormatUnsupportedError(
                "cannot append to a migrated session until it is republished "
                f"as current-version v4 (session {self.session_id!r}; "
                f"raw log: {self.storage_path})"
            )
        if event.seq == 0:
            event.seq = len(self.events) + 1
        self.events.append(event)
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "a", encoding="utf-8") as f:
                try:
                    _lock_exclusive(f)
                except (ImportError, OSError, AttributeError):
                    pass
                try:
                    if needs_header:
                        header: dict[str, Any] = {
                            "version": CURRENT_FORMAT_VERSION,
                            "id": self.session_id,
                            "createdAt": int(time.time() * 1000),
                            "isSeeded": False,
                            "delegationDepth": 0,
                        }
                        physical = SESSION_FORMAT_CATALOG.encode_current_header(
                            header, 0
                        )
                        f.write(json.dumps(physical) + "\n")
                        self._header = header
                    row = encode_envelope_event(
                        {
                            "type": event.type.value,
                            "seq": event.seq,
                            "time": int(event.timestamp * 1000),
                            "data": event.payload,
                            **({"ignorable": True} if event.ignorable else {}),
                        }
                    )
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                finally:
                    try:
                        _unlock(f)
                    except (ImportError, OSError, AttributeError):
                        pass

    def derive_messages(self) -> list[dict[str, Any]]:
        """Project current model-visible messages from event log."""
        return derive_messages(self.events)

    @classmethod
    def load(cls, storage_path: Path) -> Session:
        """Load session from a JSONL file.

        Refuses by default: a newer-version log, a malformed header, a
        corrupt row, or an unmigratable history raises instead of being
        silently truncated or repaired. Missing or empty files still load
        as an empty (phantom) session; that ownership belongs to the
        persistence rewrite.
        """
        session_id = storage_path.stem
        header: dict[str, Any] | None = None
        events: list[SessionEvent] = []
        on_disk_current = True
        if storage_path.exists():
            lines = [
                line.strip()
                for line in storage_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if lines:
                header, events, on_disk_current = _load_lines(lines, storage_path)
                session_id = header["id"]
        sess = cls(session_id=session_id, storage_path=storage_path)
        sess.events = events
        sess._header = header
        sess._on_disk_current = on_disk_current
        return sess


def _parse_row(line: str, row_index: int, storage_path: Path) -> dict[str, Any]:
    """Parse one JSONL row; corrupt JSON is corruption, never truncation."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise SessionPersistenceCorruptionError(
            f"corrupt session row {row_index} in {storage_path}: {error}"
        ) from error
    if not isinstance(row, dict):
        raise SessionPersistenceCorruptionError(
            f"corrupt session row {row_index} in {storage_path}: "
            "row must be a JSON object"
        )
    return row


def _to_session_event(
    event: dict[str, Any], session_id: str, storage_path: Path
) -> SessionEvent | None:
    """Map one v4 logical event to a SessionEvent.

    Returns ``None`` for ignorable events outside this build's vocabulary.
    Unknown non-ignorable events are refused. The vocabulary covers every
    format-admitted v4 type, so the migration's own validated output loads;
    per-type payload and lifecycle validation belongs to later work.
    """
    event_type = event.get("type")
    try:
        mapped_type = EventType(event_type)
    except ValueError:
        if event.get("ignorable") is True:
            return None
        raise SessionFormatUnsupportedError(
            f"session {session_id!r} contains unknown event type "
            f"{json.dumps(event_type)}; this build cannot load it "
            f"(raw log: {storage_path})"
        ) from None
    seq = require_count(event.get("seq"), f"session event {event_type} seq")
    data = event.get("data")
    return SessionEvent(
        id=f"{session_id}:{seq}",
        session_id=session_id,
        type=mapped_type,
        timestamp=require_safe_integer(
            event.get("time"), f"session event {event_type} time"
        )
        / 1000.0,
        payload=data if is_json_object(data) else {"value": data},
        ignorable=event.get("ignorable") is True,
        seq=seq,
    )


def _load_current_rows(
    lines: list[str], session_id: str, storage_path: Path
) -> list[SessionEvent]:
    """Envelope-only read of a current-version file's event rows."""
    events: list[SessionEvent] = []
    for index, line in enumerate(lines[1:], start=1):
        row = _parse_row(line, index, storage_path)
        try:
            decoded = decode_envelope_row(row, index, f"session {session_id}")
        except SessionFormatError as error:
            raise SessionPersistenceCorruptionError(str(error), error) from error
        mapped = _to_session_event(decoded, session_id, storage_path)
        if mapped is not None:
            events.append(mapped)
    return events


def _load_migrated_rows(
    lines: list[str],
    header_value: dict[str, Any],
    session_id: str,
    storage_path: Path,
) -> list[SessionEvent]:
    """Restore a historical file through the catalog into v4 events."""
    catalog = create_session_format_catalog_with_children([])
    try:
        restore = catalog.create_restore(
            header_value, validation="transformed", recovery="strict"
        )
        for index, line in enumerate(lines[1:], start=1):
            restore.decode_row(_parse_row(line, index, storage_path))
        artifact = restore.finish()
    except SessionFormatUnsupportedMigrationError as error:
        raise SessionFormatUnsupportedError(
            f"{error} (raw log: {storage_path})"
        ) from error
    except SessionFormatError as error:
        raise SessionPersistenceCorruptionError(str(error), error) from error
    events: list[SessionEvent] = []
    for event in artifact.events:
        mapped = _to_session_event(event, session_id, storage_path)
        if mapped is not None:
            events.append(mapped)
    return events


def _classify_header_line(
    line: str, storage_path: Path
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Classify one stored log's header line without decoding event rows.

    Returns the physical header value, the logical header, and whether the
    log is already current-version v4. Unsupported and malformed headers
    raise the same errors Session.load raises for them.
    """
    try:
        header_value = json.loads(line)
    except json.JSONDecodeError as error:
        raise SessionPersistenceCorruptionError(
            f"corrupt session header in {storage_path}: {error}"
        ) from error
    read = SESSION_FORMAT_CATALOG.read_header(header_value)
    if read.status == "unsupported":
        if (
            read.stored_version is not None
            and read.stored_version > CURRENT_FORMAT_VERSION
        ):
            header_id = (
                header_value.get("id")
                if is_json_object(header_value)
                else storage_path.stem
            )
            message = session_format_version_refusal(
                str(header_id), read.stored_version
            )
        else:
            # Unreachable with the shipped catalog: a header that decodes
            # cannot fail a later edge validation (all edges validate the
            # same header fields), and the missing-codec path is excluded
            # at catalog construction.
            message = read.reason or "unsupported session format"  # pragma: no cover
        raise SessionFormatUnsupportedError(f"{message} (raw log: {storage_path})")
    if read.status == "malformed":
        raise SessionFormatError(read.reason or "malformed session header")
    assert read.header is not None
    return header_value, read.header, read.status == "current"


def _on_disk_header(storage_path: Path) -> tuple[dict[str, Any] | None, bool]:
    """Classify the header line of an existing stored log, if it has one.

    Returns the logical header and whether the log is current-version v4.
    A file with no non-empty line has no header yet and reports
    ``(None, True)``; it is treated as a fresh log. Unsupported and
    malformed headers raise the same errors Session.load raises.
    """
    with open(storage_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                _, header, is_current = _classify_header_line(line, storage_path)
                return header, is_current
    return None, True


def _load_lines(
    lines: list[str], storage_path: Path
) -> tuple[dict[str, Any], list[SessionEvent], bool]:
    """Classify one stored log's header, then decode or migrate its rows.

    The third return value reports whether the on-disk log is already
    current-version v4; anything else is a migrated view that must not be
    appended to until it is republished.
    """
    header_value, header, on_disk_current = _classify_header_line(
        lines[0], storage_path
    )
    session_id = header["id"]
    if on_disk_current:
        events = _load_current_rows(lines, session_id, storage_path)
    else:
        events = _load_migrated_rows(lines, header_value, session_id, storage_path)
    return header, events, on_disk_current
