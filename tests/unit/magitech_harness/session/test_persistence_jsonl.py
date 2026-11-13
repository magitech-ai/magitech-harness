"""Unit tests for the plain-JSONL layout, generation resolution, and lease."""

import dataclasses
import errno
import json
import logging
import os
import stat as stat_module
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from anima import Context

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.format import (
    SESSION_FORMAT_CATALOG,
    SessionFormatError,
    SessionFormatUnsupportedError,
    SessionPersistenceCorruptionError,
)
from magitech_harness.session.persistence_jsonl import (
    JsonlLiveTracker,
    JsonlSessionHandle,
    JsonlSessionPersistence,
    KernelLease,
    ReadResult,
    ScanResult,
    SessionAlreadyExistsError,
    SessionAlreadyOwnedError,
    SessionHandleClosedError,
    SessionOwnershipLostError,
    SessionPersistenceNotFoundError,
    SessionPersistenceSnapshot,
    SessionReadOnlyError,
    _acquire_posix,
    _migration_event_row,
    encode_segment,
    find_session_dir,
    generation_filename,
    parse_generation_filename,
    project_dir,
    project_key,
    read_generation,
    resolve_current_generation,
    scan_log_bytes,
    session_dir,
    verify_current_generation,
)


def test_errors_carry_identity() -> None:
    """Canonical errors keep the frozen names and signatures."""
    read_only = SessionReadOnlyError("s-1", "append")
    assert read_only.session_id == "s-1"
    assert read_only.operation == "append"
    assert "s-1" in str(read_only) and "append" in str(read_only)
    closed = SessionHandleClosedError("s-1", "flush")
    assert closed.session_id == "s-1"
    assert closed.operation == "flush"
    assert "s-1" in str(closed) and "flush" in str(closed)
    assert str(SessionPersistenceNotFoundError("missing"))
    assert str(SessionAlreadyExistsError("exists"))
    assert str(SessionAlreadyOwnedError("owned"))
    # Declared, never raised by any shipped backend.
    assert str(SessionOwnershipLostError("gone"))


def test_encode_segment_special_cases() -> None:
    """Dot segments are escaped; empty input is refused."""
    assert encode_segment(".") == "~002E"
    assert encode_segment("..") == "~002E~002E"
    with pytest.raises(ValueError):
        encode_segment("")


def test_encode_segment_injective_no_traversal() -> None:
    """Distinct ids encode distinctly, with no separator left behind."""
    raw_ids = [
        "a",
        "b",
        ".",
        "..",
        "a/b",
        "a\\b",
        "a~b",
        "a b",
        "a:b",
        "a~002Fb",
        "aAb",
        "a\u00e9b",
        "\U0001f600",
        "a.b",
        "a..b",
        " session-1",
        "session-1",
        "\ud800",
    ]
    encoded = [encode_segment(raw) for raw in raw_ids]
    assert len(set(encoded)) == len(raw_ids)
    for segment in encoded:
        assert "/" not in segment and "\\" not in segment
        assert segment not in (".", "..")
    assert encode_segment("a/b") == "a~002Fb"
    assert encode_segment("a~b") == "a~007Eb"
    assert encode_segment("\U0001f600") == "~D83D~DE00"
    assert encode_segment("session-1") == "session-1"


def test_project_key_separators_drive_letters_and_lossy() -> None:
    """Separators collapse, drive letters become dashes, collisions lose."""
    assert project_key("C:\\Users\\x") == "--C-Users-x--"
    assert project_key("/a//b") == "--a-b--"
    assert project_key("/a") == "--a--"
    assert project_key("/") == "--root--"
    assert project_key("a/b") == project_key("a\\b")
    assert project_key("a b") == "--a~0020b--"
    with pytest.raises(ValueError):
        project_key("")


def test_project_key_truncates_slug_to_251() -> None:
    """The slug is cut at 251 chars, so long paths collide by truncation."""
    long_cwd = "/a" * 200
    key = project_key(long_cwd)
    assert len(key) == 255
    assert key.startswith("--") and key.endswith("--")
    assert project_key("x" * 300) == project_key("x" * 300 + "y")


def test_layout_dirs(tmp_path: Path) -> None:
    """Project/session dirs follow the no-cwd and slug rules."""
    root = tmp_path / "root"
    assert project_dir(root, None) == root / "_no-cwd"
    assert project_dir(root, "/work/proj") == root / "--work-proj--"
    assert session_dir(root, None, "s-1") == root / "_no-cwd" / "s-1"
    assert session_dir(root, "/work/proj", "a/b") == (
        root / "--work-proj--" / "a~002Fb"
    )


def test_find_session_dir_scans_projects(tmp_path: Path) -> None:
    """The session dir is found without knowing its working directory."""
    root = tmp_path / "root"
    (root / "stray.txt").parent.mkdir(parents=True)
    (root / "stray.txt").write_text("not a project", encoding="utf-8")
    found = root / "--work-proj--" / "s-1"
    found.mkdir(parents=True)
    (root / "--other--").mkdir()
    assert find_session_dir(root, "s-1") == found
    assert find_session_dir(root, "missing") is None
    assert find_session_dir(tmp_path / "nope", "s-1") is None


def test_generation_filename_wrappers() -> None:
    """Generation names round-trip through the format layer."""
    for version in (0, 1, 2, 3, 4):
        assert parse_generation_filename(generation_filename(version)) == version
    assert generation_filename(0) == "session.jsonl"
    assert generation_filename(4) == "session.v4.jsonl"
    assert parse_generation_filename("session.v01.jsonl") is None
    assert parse_generation_filename("SESSION.JSONL") is None
    assert parse_generation_filename("session.v0.jsonl") is None


def _write(session_path: Path, name: str) -> Path:
    target = session_path / name
    target.write_text("{}", encoding="utf-8")
    return target


def test_resolve_current_generation_highest_wins(tmp_path: Path) -> None:
    """The highest version present is the current generation."""
    session_path = tmp_path / "s-1"
    session_path.mkdir()
    _write(session_path, "session.jsonl")
    _write(session_path, "session.v2.jsonl")
    newest = _write(session_path, "session.v4.jsonl")
    assert resolve_current_generation(session_path) == (4, newest)


def test_resolve_current_generation_ignores_noncanonical(tmp_path: Path) -> None:
    """Noncanonical, temp, uppercase, and leading-zero names are ignored."""
    session_path = tmp_path / "s-1"
    session_path.mkdir()
    for name in (
        "SESSION.JSONL",
        "session.v02.jsonl",
        "session.v0.jsonl",
        "session.jsonl.tmp",
        "session.v1.jsonl.bak",
        "session.vX.jsonl",
    ):
        _write(session_path, name)
    assert resolve_current_generation(session_path) is None
    (session_path / "session.v9.jsonl").mkdir()
    assert resolve_current_generation(session_path) is None


def test_resolve_current_generation_zstd_is_hard_error(tmp_path: Path) -> None:
    """A compressed artifact in the session dir fails loud, naming the file."""
    session_path = tmp_path / "s-1"
    session_path.mkdir()
    _write(session_path, "session.jsonl")
    zstd = _write(session_path, "session.jsonl.zstd")
    with pytest.raises(SessionPersistenceCorruptionError) as excinfo:
        resolve_current_generation(session_path)
    assert "session.jsonl.zstd" in str(excinfo.value)
    assert str(zstd) in str(excinfo.value)


def test_resolve_current_generation_empty_or_missing(tmp_path: Path) -> None:
    """An empty or missing session directory has no current generation."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_current_generation(empty) is None
    assert resolve_current_generation(tmp_path / "missing") is None


def _session_path(tmp_path: Path) -> Path:
    session_path = tmp_path / "s-1"
    session_path.mkdir()
    return session_path


def test_lease_second_acquire_in_process_is_owned(tmp_path: Path) -> None:
    """A second acquire in the same process contends and fails."""
    session_path = _session_path(tmp_path)
    lease = KernelLease.acquire(session_path, "s-1")
    try:
        with pytest.raises(SessionAlreadyOwnedError):
            KernelLease.acquire(session_path, "s-1")
    finally:
        lease.release()


def test_lease_release_then_reacquire_and_file_remains(tmp_path: Path) -> None:
    """Release frees the lease; the lock file is never removed."""
    session_path = _session_path(tmp_path)
    lease = KernelLease.acquire(session_path, "s-1")
    lock_path = session_path / "session.lock"
    assert lock_path.is_file()
    if sys.platform != "win32":
        # Windows ignores permission bits on creation.
        assert stat_module.S_IMODE(lock_path.stat().st_mode) == 0o600
    lease.release()
    assert lock_path.is_file()
    lease2 = KernelLease.acquire(session_path, "s-1")
    lease2.release()
    assert lock_path.is_file()


def test_lease_release_is_idempotent_and_context_managed(
    tmp_path: Path,
) -> None:
    """Double release is a no-op; the context manager releases on exit."""
    session_path = _session_path(tmp_path)
    lease = KernelLease.acquire(session_path, "s-1")
    lease.release()
    lease.release()
    with (
        KernelLease.acquire(session_path, "s-1"),
        pytest.raises(SessionAlreadyOwnedError),
    ):
        KernelLease.acquire(session_path, "s-1")
    KernelLease.acquire(session_path, "s-1").release()


def test_lease_acquire_on_missing_dir_fails_loud(tmp_path: Path) -> None:
    """Acquiring a lease for a missing session dir raises, never creates."""
    with pytest.raises(FileNotFoundError):
        KernelLease.acquire(tmp_path / "missing", "missing")


def test_plain_reads_never_touch_the_lock(tmp_path: Path) -> None:
    """A plain file read creates no lock file."""
    session_path = _session_path(tmp_path)
    log = session_path / "session.jsonl"
    log.write_text("{}\n", encoding="utf-8")
    assert log.read_text(encoding="utf-8") == "{}\n"
    assert not (session_path / "session.lock").exists()


def _stat_fake(lock_path: Path, mismatches: int, *, raise_once: bool = False):
    """Fake os.stat: mismatch (or raise) the first N lock-path lookups."""
    real_stat = os.stat
    calls = {"n": 0}

    def fake_stat(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = real_stat(path, *args, **kwargs)
        if os.fspath(path) == os.fspath(lock_path) and calls["n"] < mismatches:
            calls["n"] += 1
            if raise_once:
                raise FileNotFoundError("lock file gone")
            fields = tuple(result)
            return os.stat_result((fields[0], fields[1] + 1000003, *fields[2:]))
        return result

    return fake_stat


def test_lease_inode_mismatch_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stale inode check retries; the next attempt takes the lease."""
    _patch_posix(monkeypatch, _FakeFcntl())
    session_path = _session_path(tmp_path)
    lock_path = session_path / "session.lock"
    monkeypatch.setattr(os, "stat", _stat_fake(lock_path, 1))
    fd = _acquire_posix(lock_path, "s-1")
    os.close(fd)


def test_lease_inode_mismatch_exhausted_is_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock file replaced on every attempt is reported as owned."""
    _patch_posix(monkeypatch, _FakeFcntl())
    session_path = _session_path(tmp_path)
    lock_path = session_path / "session.lock"
    monkeypatch.setattr(os, "stat", _stat_fake(lock_path, 99))
    with pytest.raises(SessionAlreadyOwnedError):
        _acquire_posix(lock_path, "s-1")


def test_lease_vanished_lock_file_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock file that vanishes mid-check retries instead of failing."""
    _patch_posix(monkeypatch, _FakeFcntl())
    session_path = _session_path(tmp_path)
    lock_path = session_path / "session.lock"
    monkeypatch.setattr(os, "stat", _stat_fake(lock_path, 1, raise_once=True))
    fd = _acquire_posix(lock_path, "s-1")
    os.close(fd)


def test_lease_posix_lock_error_other_than_contention_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-contention flock failure fails loud, never as owned."""
    fake = _FakeFcntl(OSError(errno.EIO, "flock boom"))
    _patch_posix(monkeypatch, fake)
    with pytest.raises(OSError, match="flock boom"):
        _acquire_posix(_session_path(tmp_path) / "session.lock", "s-1")


def test_lease_posix_contention_is_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A POSIX lock conflict reports the session as already owned."""
    fake = _FakeFcntl(OSError(errno.EAGAIN, "locked"))
    _patch_posix(monkeypatch, fake)
    with pytest.raises(SessionAlreadyOwnedError):
        _acquire_posix(_session_path(tmp_path) / "session.lock", "s-1")


def test_lease_posix_refuses_windows_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The POSIX lease refuses to run when the platform is Windows."""
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="POSIX lease"):
        _acquire_posix(_session_path(tmp_path) / "session.lock", "s-1")


class _FakeMsvcrt:
    """Stand-in for the Windows byte-range locking module."""

    LK_NBLCK = 1
    LK_UNLCK = 0

    def __init__(self, error: OSError | None = None) -> None:
        self.calls: list[tuple[int, int, int]] = []
        self.error = error

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((fd, mode, nbytes))
        if self.error is not None:
            raise self.error


def _patch_windows(monkeypatch: pytest.MonkeyPatch, fake: _FakeMsvcrt) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", fake)


class _FakeFcntl:
    """Stand-in for the POSIX flock module."""

    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    def __init__(self, error: OSError | None = None) -> None:
        self.calls: list[tuple[int, int]] = []
        self.error = error

    def flock(self, fd: int, operation: int) -> None:
        self.calls.append((fd, operation))
        if self.error is not None:
            raise self.error


def _patch_posix(monkeypatch: pytest.MonkeyPatch, fake: _FakeFcntl) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "fcntl", fake)


def test_lease_windows_acquire_and_release(tmp_path: Path, monkeypatch) -> None:
    """The Windows path locks byte 0 non-blocking and unlocks on release."""
    fake = _FakeMsvcrt()
    _patch_windows(monkeypatch, fake)
    session_path = _session_path(tmp_path)
    lease = KernelLease.acquire(session_path, "s-1")
    assert fake.calls == [(lease._fd, _FakeMsvcrt.LK_NBLCK, 1)]
    lease.release()
    assert fake.calls[-1][1] == _FakeMsvcrt.LK_UNLCK
    assert (session_path / "session.lock").is_file()


def test_lease_windows_contention_is_owned(tmp_path: Path, monkeypatch) -> None:
    """A Windows lock conflict reports the session as already owned."""
    fake = _FakeMsvcrt(OSError(errno.EACCES, "locked"))
    _patch_windows(monkeypatch, fake)
    with pytest.raises(SessionAlreadyOwnedError):
        KernelLease.acquire(_session_path(tmp_path), "s-1")


def test_lease_windows_other_lock_error_propagates(tmp_path: Path, monkeypatch) -> None:
    """A non-contention Windows lock failure fails loud, never as owned."""
    fake = _FakeMsvcrt(OSError(errno.EIO, "locking boom"))
    _patch_windows(monkeypatch, fake)
    with pytest.raises(OSError, match="locking boom"):
        KernelLease.acquire(_session_path(tmp_path), "s-1")


def test_lease_posix_acquire_and_release(tmp_path: Path, monkeypatch) -> None:
    """The POSIX path locks non-blocking and unlocks on release."""
    fake = _FakeFcntl()
    _patch_posix(monkeypatch, fake)
    session_path = _session_path(tmp_path)
    lease = KernelLease.acquire(session_path, "s-1")
    assert fake.calls == [(lease._fd, _FakeFcntl.LOCK_EX | _FakeFcntl.LOCK_NB)]
    lease.release()
    assert fake.calls[-1][1] == _FakeFcntl.LOCK_UN
    assert (session_path / "session.lock").is_file()


# ---------------------------------------------------------------------------
# Slice 2a: byte-level scan, generation read, verification read
# ---------------------------------------------------------------------------


def test_scan_log_bytes_complete_and_torn() -> None:
    """Complete lines split on newlines; the unterminated tail is torn."""
    data = b'{"a":1}\n{"b":2}\n{"c":3}'
    result = scan_log_bytes(data)
    assert isinstance(result, ScanResult)
    assert result.lines == [b'{"a":1}', b'{"b":2}']
    assert result.torn_fragment == b'{"c":3}'
    assert result.torn_truncate_to == data.index(b'{"c":3}')


def test_scan_log_bytes_torn_fragment_even_when_parseable() -> None:
    """A final record without a newline is torn even if it parses as JSON."""
    result = scan_log_bytes(b'{}\n{"type":"turn/end"}\n{"type":"turn/start"}')
    assert result.lines == [b"{}", b'{"type":"turn/end"}']
    assert result.torn_fragment == b'{"type":"turn/start"}'
    assert result.torn_truncate_to == 23


def test_scan_log_bytes_trailing_newline_has_no_torn() -> None:
    """A file ending exactly with a newline has no torn fragment."""
    result = scan_log_bytes(b'{}\n{"a":1}\n')
    assert result.lines == [b"{}", b'{"a":1}']
    assert result.torn_truncate_to is None
    assert result.torn_fragment == b""


def test_scan_log_bytes_empty_input() -> None:
    """Empty input has no lines and no torn fragment."""
    result = scan_log_bytes(b"")
    assert result.lines == []
    assert result.torn_truncate_to is None
    assert result.torn_fragment == b""


def _dump(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _v4_header(session_id: str = "s-1", **extra: object) -> dict:
    header: dict = {
        "version": 4,
        "id": session_id,
        "createdAt": 1000,
        "isSeeded": False,
        "delegationDepth": 0,
    }
    header.update(extra)
    return header


def _v4_event(event_type: str, seq: int, time: int, data: dict) -> dict:
    return {"type": event_type, "seq": seq, "time": time, "data": data}


def _write_v4_log(
    path: Path,
    header: dict,
    rows: list[dict | str],
    *,
    cut: int = 0,
    trailing_newline: bool = True,
) -> Path:
    """Write a current-version log, encoding good rows via the format layer."""
    lines = [_dump(SESSION_FORMAT_CATALOG.encode_current_header(header, cut))]
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
        else:
            lines.append(_dump(SESSION_FORMAT_CATALOG.encode_current_event(row)))
    text = "\n".join(lines)
    if trailing_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _good_turn(session_id: str = "s-1") -> tuple[dict, list[dict]]:
    header = _v4_header(session_id)
    rows = [
        _v4_event("turn/start", 0, 1000, {"turn": 1}),
        _v4_event("step/start", 1, 1001, {"turn": 1, "step": 1}),
        _v4_event("turn/end", 2, 1002, {"turn": 1, "reason": {"kind": "done"}}),
    ]
    return header, rows


def test_read_generation_current_v4(tmp_path: Path) -> None:
    """A current log reads header, events, cut, status, and file identity."""
    log = tmp_path / "session.v4.jsonl"
    header, rows = _good_turn()
    _write_v4_log(log, header, rows)
    result = read_generation(log)
    assert isinstance(result, ReadResult)
    assert result.header["id"] == "s-1"
    assert result.header["version"] == 4
    assert [event.type for event in result.events] == [
        "turn/start",
        "step/start",
        "turn/end",
    ]
    assert [event.seq for event in result.events] == [0, 1, 2]
    assert result.inherited_event_count == 0
    assert result.format_status == "current"
    assert result.torn_truncate_to is None
    assert result.recovered_tail == []
    info = log.stat()
    assert result.revision_base == (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def test_read_generation_current_v4_seeded_inherited_cut(tmp_path: Path) -> None:
    """The inherited cut comes from the last tagged end-seed marker."""
    log = tmp_path / "session.v4.jsonl"
    header = _v4_header(isSeeded=True)
    rows = [
        _v4_event("turn/start", 0, 1000, {"turn": 1}),
        _v4_event("session/end-seed", 1, 1001, {"inherited": True}),
        _v4_event("turn/end", 2, 1002, {"turn": 1, "reason": {"kind": "done"}}),
    ]
    _write_v4_log(log, header, rows, cut=1)
    result = read_generation(log)
    assert result.inherited_event_count == 1
    assert result.format_status == "current"


def test_read_generation_header_only_log(tmp_path: Path) -> None:
    """A log with only a header reads as zero events."""
    log = tmp_path / "session.v4.jsonl"
    _write_v4_log(log, _v4_header(), [])
    result = read_generation(log)
    assert result.events == []
    assert result.inherited_event_count == 0
    assert result.format_status == "current"


def test_verify_current_generation_accepts_clean_log(tmp_path: Path) -> None:
    """The verification read returns a clean current generation unchanged."""
    log = tmp_path / "session.v4.jsonl"
    header, rows = _good_turn()
    _write_v4_log(log, header, rows)
    result = verify_current_generation(log)
    assert result.format_status == "current"
    assert len(result.events) == 3


def _bad_envelope_row() -> str:
    """A row that parses but lacks the required envelope data member."""
    return _dump({"type": "step/start", "seq": 1, "time": 1001})


def test_read_generation_recoverable_rethrows_at_turn_end(tmp_path: Path) -> None:
    """Recoverable: first bad row recorded, later rows suppressed, re-thrown at turn/end."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            _bad_envelope_row(),
            _v4_event("step/start", 2, 1002, {"turn": 1, "step": 1}),
            _v4_event("turn/end", 3, 1003, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    with pytest.raises(
        SessionPersistenceCorruptionError, match="lacks required field data"
    ):
        read_generation(log)


def test_read_generation_recoverable_corrupt_tail_raises_at_end(
    tmp_path: Path,
) -> None:
    """Recoverable: a corrupt tail with no following turn/end raises at end of read."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            "{not json",
            _v4_event("step/start", 2, 1002, {"turn": 1, "step": 1}),
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="row 2"):
        read_generation(log)


def test_read_generation_recoverable_two_bad_rows_keeps_first(tmp_path: Path) -> None:
    """Recoverable: the first bad row's issue is the one re-thrown, not the second's."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            _bad_envelope_row(),
            "{oops",
            _v4_event("turn/end", 3, 1003, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    with pytest.raises(
        SessionPersistenceCorruptionError, match="lacks required field data"
    ):
        read_generation(log)


def test_read_generation_recoverable_seq_gap_rethrows_at_turn_end(
    tmp_path: Path,
) -> None:
    """Recoverable: a seq gap is recorded like any other bad row."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            _v4_event("step/start", 9, 1001, {"turn": 1, "step": 1}),
            _v4_event("turn/end", 2, 1002, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="seq gap"):
        read_generation(log)


def test_read_generation_strict_bad_envelope_raises_at_once(tmp_path: Path) -> None:
    """Strict: the first bad row raises immediately."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            _bad_envelope_row(),
            _v4_event("turn/end", 2, 1002, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    with pytest.raises(
        SessionPersistenceCorruptionError, match="lacks required field data"
    ):
        read_generation(log, recovery="strict")


def test_read_generation_strict_seq_gap_raises(tmp_path: Path) -> None:
    """Strict: a seq gap raises, naming the expected and actual seq."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            _v4_event("step/start", 5, 1001, {"turn": 1, "step": 1}),
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="seq gap"):
        read_generation(log, recovery="strict")


def test_read_generation_strict_non_integer_seq_raises(tmp_path: Path) -> None:
    """Strict: a row whose seq passes the gap check but is not an integer raises."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            '{"type":"step/start","seq":1.0,"time":1001,"data":{"turn":1,"step":1}}',
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="safe integer"):
        read_generation(log, recovery="strict")


def test_read_generation_unparsable_json_names_line(tmp_path: Path) -> None:
    """Unparsable committed JSON is corruption naming the line number."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            "{bad json",
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="row 2"):
        read_generation(log, recovery="strict")


def test_read_generation_newer_version_refused_with_upgrade_text(
    tmp_path: Path,
) -> None:
    """A newer-version header is refused as unsupported, never as corrupt."""
    log = tmp_path / "session.v5.jsonl"
    log.write_text(
        _dump(
            {
                "type": "session",
                "version": 5,
                "id": "s-1",
                "createdAt": 1000,
                "isSeeded": False,
                "delegationDepth": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionFormatUnsupportedError, match="upgrade the harness"):
        read_generation(log)


def test_read_generation_malformed_header_refused_as_malformed(
    tmp_path: Path,
) -> None:
    """A malformed header is refused as malformed, not as unsupported."""
    log = tmp_path / "session.v4.jsonl"
    log.write_text(
        _dump(
            {
                "type": "session",
                "version": 4,
                "id": 3,
                "createdAt": 1000,
                "isSeeded": False,
                "delegationDepth": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SessionFormatError, match="must be a string"):
        read_generation(log)


def test_read_generation_unknown_type_refused(tmp_path: Path) -> None:
    """An unknown non-ignorable event type is unsupported, never corruption."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            _v4_event("bogus/thing", 1, 1001, {"turn": 1}),
        ],
    )
    with pytest.raises(SessionFormatUnsupportedError, match="unknown event type"):
        read_generation(log)


def test_read_generation_retired_type_refused_as_unsupported(tmp_path: Path) -> None:
    """A retired row fails admission outright, even in recoverable mode."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            # Hand-built: the encoder itself refuses retired rows.
            '{"type":"tool/code-dispatch-start","seq":1,"time":1001,"data":{"turn":1}}',
        ],
    )
    with pytest.raises(SessionFormatUnsupportedError, match="retired event type"):
        read_generation(log)


def _write_v0_log(path: Path, rows: list[dict | str], session_id: str = "s-1") -> Path:
    """Hand-build a minimal v0 log file (no encoder exists for v0)."""
    header = {
        "type": "session",
        "version": 0,
        "id": session_id,
        "createdAt": 1000,
        "delegationDepth": 0,
    }
    lines = [_dump(header)]
    lines.extend(row if isinstance(row, str) else _dump(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _v0_turn() -> list[dict]:
    return [
        _v4_event("turn/start", 0, 1000, {"turn": 1, "trigger": {"kind": "user"}}),
        _v4_event("step/start", 1, 1001, {"turn": 1, "step": 1}),
        _v4_event("step/end", 2, 1002, {"turn": 1, "step": 1}),
        _v4_event("turn/end", 3, 1003, {"turn": 1, "reason": {"kind": "done"}}),
    ]


def test_read_generation_historical_v0_is_migration_required(tmp_path: Path) -> None:
    """A v0 log reads as migration-required with correctly migrated events."""
    log = tmp_path / "session.jsonl"
    _write_v0_log(log, _v0_turn())
    result = read_generation(log)
    assert result.format_status == "migration-required"
    assert [event.type for event in result.events] == [
        "turn/start",
        "step/start",
        "system/message",
        "step/end",
        "turn/end",
    ]
    assert [event.seq for event in result.events] == [0, 1, 2, 3, 4]
    assert result.inherited_event_count == 0


def test_read_generation_historical_v0_strict_read(tmp_path: Path) -> None:
    """A clean v0 log also reads under strict recovery."""
    log = tmp_path / "session.jsonl"
    _write_v0_log(log, _v0_turn())
    result = read_generation(log, recovery="strict")
    assert result.format_status == "migration-required"
    assert len(result.events) == 5


def test_read_generation_historical_unknown_type_refused(tmp_path: Path) -> None:
    """An unknown historical event type is unsupported, never corruption."""
    log = tmp_path / "session.jsonl"
    _write_v0_log(
        log,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1, "trigger": {"kind": "user"}}),
            _v4_event("bogus/thing", 1, 1001, {"turn": 1}),
        ],
    )
    with pytest.raises(
        SessionFormatUnsupportedError, match="unknown historical event type"
    ):
        read_generation(log)


def test_read_generation_historical_recoverable_rethrows_at_turn_end(
    tmp_path: Path,
) -> None:
    """Historical recoverable: a bad row re-throws at the next turn/end row."""
    log = tmp_path / "session.jsonl"
    _write_v0_log(
        log,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1, "trigger": {"kind": "user"}}),
            _v4_event("step/start", 1, 1001, {"turn": 1, "step": 1}),
            {"type": "step/end", "seq": 2, "time": 1002},
            _v4_event("turn/end", 3, 1003, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    with pytest.raises(
        SessionPersistenceCorruptionError, match="lacks required field data"
    ):
        read_generation(log)


def test_read_generation_historical_recoverable_swallowed_issue_raises_at_end(
    tmp_path: Path,
) -> None:
    """Historical recoverable: a bad row with no following turn/end still raises."""
    log = tmp_path / "session.jsonl"
    _write_v0_log(
        log,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1, "trigger": {"kind": "user"}}),
            _v4_event("step/start", 1, 1001, {"turn": 1, "step": 1}),
            {"type": "step/end", "seq": 2, "time": 1002},
            _v4_event("turn/start", 3, 1003, {"turn": 2, "trigger": {"kind": "user"}}),
        ],
    )
    with pytest.raises(
        SessionPersistenceCorruptionError, match="lacks required field data"
    ):
        read_generation(log)


def test_read_generation_historical_unparsable_row_rethrows_at_turn_end(
    tmp_path: Path,
) -> None:
    """Historical recoverable: an unparsable row re-throws at the next turn/end."""
    log = tmp_path / "session.jsonl"
    _write_v0_log(
        log,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1, "trigger": {"kind": "user"}}),
            "{bad json",
            _v4_event("turn/end", 2, 1002, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="row 2"):
        read_generation(log)


def test_read_generation_historical_unparsable_tail_raises_at_end(
    tmp_path: Path,
) -> None:
    """Historical recoverable: an unparsable tail with no turn/end raises at end."""
    log = tmp_path / "session.jsonl"
    _write_v0_log(
        log,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1, "trigger": {"kind": "user"}}),
            "{bad json",
            _v4_event("step/start", 2, 1002, {"turn": 1, "step": 1}),
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="row 2"):
        read_generation(log)


def test_verify_current_generation_rejects_torn_tail(tmp_path: Path) -> None:
    """A torn tail is rejected on verification but reads as a prefix otherwise."""
    log = tmp_path / "session.v4.jsonl"
    header, rows = _good_turn()
    lines = [_dump(SESSION_FORMAT_CATALOG.encode_current_header(header, 0))]
    lines.extend(
        _dump(SESSION_FORMAT_CATALOG.encode_current_event(row)) for row in rows
    )
    data = "\n".join(lines).encode("utf-8")  # no trailing newline: last row is torn
    log.write_bytes(data)
    torn_at = data.index(lines[-1].encode("utf-8"))
    plain = read_generation(log)
    assert [event.type for event in plain.events] == ["turn/start", "step/start"]
    assert plain.torn_truncate_to == torn_at
    assert plain.recovered_tail == []
    with pytest.raises(
        SessionPersistenceCorruptionError,
        match=f"torn physical tail at byte offset {torn_at}",
    ):
        verify_current_generation(log)


def test_read_generation_empty_file_is_corrupt(tmp_path: Path) -> None:
    """A file with no complete line has no header to classify."""
    log = tmp_path / "session.v4.jsonl"
    log.write_bytes(b"")
    with pytest.raises(
        SessionPersistenceCorruptionError, match="first line is not a session header"
    ):
        read_generation(log)


def test_read_generation_header_not_utf8_is_corrupt(tmp_path: Path) -> None:
    """A header line that is not valid UTF-8 is corruption, not unsupported."""
    log = tmp_path / "session.v4.jsonl"
    log.write_bytes(b"\xff\xfe\n")
    with pytest.raises(
        SessionPersistenceCorruptionError, match="corrupt session header"
    ):
        read_generation(log)


def test_read_generation_body_row_not_utf8_is_corrupt(tmp_path: Path) -> None:
    """A body row that is not valid UTF-8 is corruption naming the line."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    lines = [_dump(SESSION_FORMAT_CATALOG.encode_current_header(header, 0))]
    lines.append(
        _dump(
            SESSION_FORMAT_CATALOG.encode_current_event(
                _v4_event("turn/start", 0, 1000, {"turn": 1})
            )
        )
    )
    log.write_bytes(("\n".join(lines) + "\n").encode("utf-8") + b"\xff\n")
    with pytest.raises(SessionPersistenceCorruptionError, match="row 2"):
        read_generation(log, recovery="strict")


def test_read_generation_recoverable_non_integer_seq_raises_at_turn_end(
    tmp_path: Path,
) -> None:
    """Recoverable: a non-integer seq is recorded and re-thrown at turn/end."""
    log = tmp_path / "session.v4.jsonl"
    header, _ = _good_turn()
    _write_v4_log(
        log,
        header,
        [
            _v4_event("turn/start", 0, 1000, {"turn": 1}),
            '{"type":"step/start","seq":1.0,"time":1001,"data":{"turn":1,"step":1}}',
            _v4_event("turn/end", 2, 1002, {"turn": 1, "reason": {"kind": "done"}}),
        ],
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="safe integer"):
        read_generation(log)


# ---------------------------------------------------------------------------
# JsonlSessionHandle
# ---------------------------------------------------------------------------


def _session_event(
    session_id: str,
    event_type: str,
    seq: int,
    data: dict | None = None,
    **overrides: Any,
) -> SessionEvent:
    """One in-memory event matching a row `_v4_event` would carry."""
    fields: dict[str, Any] = {
        "id": f"{session_id}:{seq}",
        "session_id": session_id,
        "type": event_type,
        "timestamp": 1.0 + seq / 1000,
        "payload": dict(data or {}),
        "seq": seq,
    }
    fields.update(overrides)
    return SessionEvent(**fields)


def _turn_events(session_id: str = "s-1") -> list[SessionEvent]:
    """The `_good_turn` rows as in-memory events."""
    return [
        _session_event(session_id, "turn/start", 0, {"turn": 1}),
        _session_event(session_id, "step/start", 1, {"turn": 1, "step": 1}),
        _session_event(
            session_id, "turn/end", 2, {"turn": 1, "reason": {"kind": "done"}}
        ),
    ]


def _open_handle(
    tmp_path: Path,
    *,
    session_id: str = "s-1",
    events: list[SessionEvent] | None = None,
    access: str = "write",
    materialized: bool = False,
    header_written: bool = False,
    torn_truncate_to: int | None = None,
    recovered_tail: list[SessionEvent] | None = None,
    lease: KernelLease | None = None,
    release_claim: Callable[[], None] | None = None,
) -> JsonlSessionHandle:
    """Build a handle directly, as `open`/`create` would."""
    return JsonlSessionHandle(
        session_id=session_id,
        header=_v4_header(session_id),
        inherited_event_count=0,
        access=access,  # type: ignore[arg-type]
        generation_path=tmp_path / session_id / "session.v4.jsonl",
        lease=lease,
        events=events,
        torn_truncate_to=torn_truncate_to,
        recovered_tail=recovered_tail,
        materialized=materialized,
        header_written=header_written,
        release_claim=release_claim,
    )


def _materialized_handle(
    tmp_path: Path,
    *,
    session_id: str = "s-1",
    rows: list[dict] | None = None,
    **kwargs: Any,
) -> tuple[JsonlSessionHandle, Path]:
    """Write a current log, then open a handle over it with the read events."""
    header, default_rows = _good_turn(session_id)
    path = tmp_path / session_id / "session.v4.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_v4_log(path, header, rows if rows is not None else default_rows)
    result = read_generation(path, recovery="recoverable")
    handle = _open_handle(
        tmp_path,
        session_id=session_id,
        events=result.events,
        materialized=True,
        header_written=True,
        **kwargs,
    )
    return handle, path


def test_handle_properties_and_constructor_copies_events(tmp_path: Path) -> None:
    """Properties report the constructor values; the adopted log is detached."""
    adopted = _turn_events()
    handle = _open_handle(tmp_path, events=adopted)
    assert handle.id == "s-1"
    assert handle.header == _v4_header("s-1")
    assert handle.inherited_event_count == 0
    assert handle.access == "write"
    assert handle.closed is False
    adopted.append(_session_event("s-1", "turn/start", 3, {"turn": 2}))
    adopted[0].payload["turn"] = 99
    _, events = handle.read()
    assert [event.seq for event in events] == [0, 1, 2]
    assert events[0].payload == {"turn": 1}


def test_handle_read_detached_slicing_and_copy(tmp_path: Path) -> None:
    """An unmaterialized handle serves detached slices; the slice is a copy."""
    handle = _open_handle(tmp_path, events=_turn_events())
    state, events = handle.read()
    assert state == "detached"
    assert [event.seq for event in events] == [0, 1, 2]
    _, middle = handle.read(1, 1)
    assert [event.seq for event in middle] == [1]
    _, tail = handle.read(1)
    assert [event.seq for event in tail] == [1, 2]
    _, empty = handle.read(3)
    assert empty == []
    _, past = handle.read(99)
    assert past == []
    _, zero = handle.read(0, 0)
    assert zero == []
    events.append(_session_event("s-1", "turn/start", 3, {"turn": 2}))
    _, again = handle.read()
    assert len(again) == 3


def test_handle_read_shared_frozen_from_file(tmp_path: Path) -> None:
    """A materialized handle serves the stored log as shared-frozen."""
    handle, _path = _materialized_handle(tmp_path)
    state, events = handle.read()
    assert state == "shared-frozen"
    assert [event.type for event in events] == [
        "turn/start",
        "step/start",
        "turn/end",
    ]


def test_handle_read_picks_up_external_growth(tmp_path: Path) -> None:
    """A later read merges rows appended to the file by someone else."""
    handle, path = _materialized_handle(tmp_path)
    header, rows = _good_turn()
    rows.append(_v4_event("turn/start", 3, 1003, {"turn": 2}))
    _write_v4_log(path, header, rows)
    _, events = handle.read()
    assert [event.seq for event in events] == [0, 1, 2, 3]
    _, new = handle.read(3)
    assert [event.seq for event in new] == [3]


def test_handle_read_shrink_below_high_water_is_corruption(tmp_path: Path) -> None:
    """A stored log that lost a previously observed prefix is corruption."""
    handle, path = _materialized_handle(tmp_path)
    header, rows = _good_turn()
    _write_v4_log(path, header, rows[:1])
    with pytest.raises(SessionPersistenceCorruptionError):
        handle.read()


def test_handle_append_writes_header_then_rows_and_re_reads_clean(
    tmp_path: Path,
) -> None:
    """A first append on a pending handle materializes header + batch."""
    handle = _open_handle(tmp_path)
    handle.append(_turn_events())
    path = tmp_path / "s-1" / "session.v4.jsonl"
    if sys.platform != "win32":
        # Windows ignores permission bits on creation.
        assert stat_module.S_IMODE(path.parent.stat().st_mode) == 0o700
    result = read_generation(path, recovery="recoverable")
    assert [event.seq for event in result.events] == [0, 1, 2]
    verify_current_generation(path)
    state, events = handle.read()
    assert state == "shared-frozen"
    assert len(events) == 3
    handle.append([_session_event("s-1", "turn/start", 3, {"turn": 2})])
    assert len(read_generation(path, recovery="recoverable").events) == 4
    assert path.read_bytes().count(b"\n") == 5


def test_handle_append_contiguity_gap_and_duplicate_are_corruption(
    tmp_path: Path,
) -> None:
    """A batch must continue the stored log with no gap or duplicate."""
    handle = _open_handle(tmp_path, events=_turn_events())
    with pytest.raises(SessionPersistenceCorruptionError):
        handle.append([_session_event("s-1", "turn/start", 5, {"turn": 2})])
    with pytest.raises(SessionPersistenceCorruptionError):
        handle.append([_session_event("s-1", "turn/start", 2, {"turn": 2})])
    with pytest.raises(SessionPersistenceCorruptionError):
        handle.append(
            [
                _session_event("s-1", "turn/start", 3, {"turn": 2}),
                _session_event("s-1", "step/start", 5, {"turn": 2, "step": 1}),
            ]
        )
    _, events = handle.read()
    assert len(events) == 3


def test_handle_append_empty_batch_is_noop_after_guards(tmp_path: Path) -> None:
    """Guards run before the empty-batch no-op; an empty append writes nothing."""
    handle = _open_handle(tmp_path, events=_turn_events())
    handle.append([])
    assert not (tmp_path / "s-1" / "session.v4.jsonl").exists()
    _, events = handle.read()
    assert len(events) == 3
    read_only = _open_handle(tmp_path, access="read")
    with pytest.raises(SessionReadOnlyError):
        read_only.append([])
    handle.close()
    with pytest.raises(SessionHandleClosedError):
        handle.append([])


def test_handle_append_rejects_bad_events_and_log_is_unchanged(
    tmp_path: Path,
) -> None:
    """Non-JSON data and admission failures refuse the batch, log untouched."""
    handle, path = _materialized_handle(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="non-JSON-serializable"):
        handle.append(
            [_session_event("s-1", "turn/start", 3, {"turn": 2, "bad": object()})]
        )
    with pytest.raises(ValueError, match="fails v4 admission"):
        handle.append([_session_event("s-1", "system/message", 3, {"turn": 1})])
    assert path.read_bytes() == before
    _, events = handle.read()
    assert len(events) == 3


def test_handle_flush_materializes_pending_handle(tmp_path: Path) -> None:
    """Flush on a pending handle writes a durable, listable header-only log."""
    handle = _open_handle(tmp_path)
    handle.flush()
    path = tmp_path / "s-1" / "session.v4.jsonl"
    assert path.is_file()
    if sys.platform != "win32":
        # Windows ignores permission bits on creation.
        assert stat_module.S_IMODE(path.parent.stat().st_mode) == 0o700
    result = read_generation(path, recovery="recoverable")
    assert result.events == []
    assert result.header["id"] == "s-1"
    state, events = handle.read()
    assert state == "shared-frozen"
    assert events == []
    handle.flush()
    assert read_generation(path, recovery="recoverable").events == []


def test_handle_read_only_rejects_append_and_flush(tmp_path: Path) -> None:
    """A read handle serves reads but refuses mutations."""
    handle, _path = _materialized_handle(tmp_path, access="read")
    state, events = handle.read()
    assert state == "shared-frozen"
    assert len(events) == 3
    with pytest.raises(SessionReadOnlyError) as excinfo:
        handle.append([_session_event("s-1", "turn/start", 3, {"turn": 2})])
    assert excinfo.value.session_id == "s-1"
    assert excinfo.value.operation == "append"
    with pytest.raises(SessionReadOnlyError) as excinfo:
        handle.flush()
    assert excinfo.value.operation == "flush"


def test_handle_closed_rejects_ops_and_double_close_succeeds(
    tmp_path: Path,
) -> None:
    """Every operation on a closed handle names itself; close is idempotent."""
    handle, _path = _materialized_handle(tmp_path)
    handle.close()
    assert handle.closed is True
    for operation, call in (
        ("read", lambda: handle.read()),
        ("append", lambda: handle.append([])),
        ("flush", handle.flush),
    ):
        with pytest.raises(SessionHandleClosedError) as excinfo:
            call()
        assert excinfo.value.session_id == "s-1"
        assert excinfo.value.operation == operation
    handle.close()


def test_handle_close_releases_lease_and_claim(tmp_path: Path) -> None:
    """Close releases the lease and calls the claim exactly once."""
    session_path = tmp_path / "s-1"
    session_path.mkdir()
    lease = KernelLease.acquire(session_path, "s-1")
    released: list[str] = []
    handle = _open_handle(
        tmp_path, lease=lease, release_claim=lambda: released.append("claim")
    )
    handle.close()
    assert released == ["claim"]
    KernelLease.acquire(session_path, "s-1").release()
    handle.close()
    assert released == ["claim"]


def test_handle_close_combines_flush_and_claim_failures(tmp_path: Path) -> None:
    """A failed flush plus a failed claim combine; the handle stays closed."""
    blocker = tmp_path / "s-1"
    blocker.write_text("not a directory", encoding="utf-8")

    def failing_claim() -> None:
        raise RuntimeError("claim boom")

    handle = _open_handle(tmp_path, release_claim=failing_claim)
    with pytest.raises(ExceptionGroup) as excinfo:
        handle.close()
    assert handle.closed is True
    assert len(excinfo.value.exceptions) == 2
    assert any(isinstance(error, FileExistsError) for error in excinfo.value.exceptions)
    assert any(
        isinstance(error, RuntimeError) and "claim boom" in str(error)
        for error in excinfo.value.exceptions
    )
    handle.close()


def test_handle_append_repairs_torn_tail(tmp_path: Path) -> None:
    """The first append truncates torn bytes; complete events survive."""
    _handle, path = _materialized_handle(tmp_path)
    torn = b'{"type":"turn/start","seq":3'
    with open(path, "ab") as handle_file:
        handle_file.write(torn)
    result = read_generation(path, recovery="recoverable")
    assert result.torn_truncate_to is not None
    assert len(result.events) == 3
    handle = _open_handle(
        tmp_path,
        events=result.events,
        torn_truncate_to=result.torn_truncate_to,
        materialized=True,
        header_written=True,
    )
    handle.append([_session_event("s-1", "turn/start", 3, {"turn": 2})])
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert torn not in raw
    assert [event.seq for event in read_generation(path).events] == [0, 1, 2, 3]
    verify_current_generation(path)


def test_handle_append_repairs_torn_tail_with_recovered_tail(
    tmp_path: Path,
) -> None:
    """Repair rewrites recovered events before the batch lands."""
    header, rows = _good_turn()
    path = tmp_path / "s-1" / "session.v4.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_v4_log(path, header, rows[:2])
    with open(path, "ab") as handle_file:
        handle_file.write(b'{"type":"turn/end","seq":2')
    result = read_generation(path, recovery="recoverable")
    assert result.torn_truncate_to is not None
    recovered = _session_event(
        "s-1", "turn/end", 2, {"turn": 1, "reason": {"kind": "done"}}
    )
    handle = _open_handle(
        tmp_path,
        events=[*result.events, recovered],
        torn_truncate_to=result.torn_truncate_to,
        recovered_tail=[recovered],
        materialized=True,
        header_written=True,
    )
    handle.append([_session_event("s-1", "turn/start", 3, {"turn": 2})])
    reread = read_generation(path, recovery="recoverable")
    assert [(event.type, event.seq) for event in reread.events] == [
        ("turn/start", 0),
        ("step/start", 1),
        ("turn/end", 2),
        ("turn/start", 3),
    ]
    verify_current_generation(path)


def test_handle_flush_does_not_repair_torn_tail(tmp_path: Path) -> None:
    """Flush leaves a torn tail alone; the next append still repairs it."""
    _handle, path = _materialized_handle(tmp_path)
    torn = b'{"type":"turn/start","seq":3'
    with open(path, "ab") as handle_file:
        handle_file.write(torn)
    before = path.read_bytes()
    result = read_generation(path, recovery="recoverable")
    handle = _open_handle(
        tmp_path,
        events=result.events,
        torn_truncate_to=result.torn_truncate_to,
        materialized=True,
        header_written=True,
    )
    handle.flush()
    assert path.read_bytes() == before
    handle.append([_session_event("s-1", "turn/start", 3, {"turn": 2})])
    assert len(read_generation(path, recovery="recoverable").events) == 4


def test_handle_close_read_handle_skips_flush(tmp_path: Path) -> None:
    """Closing a read handle releases the claim without flushing."""
    released: list[str] = []
    handle, _path = _materialized_handle(
        tmp_path, access="read", release_claim=lambda: released.append("claim")
    )
    handle.close()
    assert released == ["claim"]
    assert handle.closed is True


def test_handle_concurrent_appends_serialize(tmp_path: Path) -> None:
    """Two threads appending at once lose no event and keep contiguity."""
    handle = _open_handle(tmp_path)
    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(5):
                for _attempt in range(50):
                    _, current = handle.read()
                    try:
                        handle.append(
                            [
                                _session_event(
                                    "s-1",
                                    "turn/start",
                                    len(current),
                                    {"turn": len(current)},
                                )
                            ]
                        )
                        break
                    except SessionPersistenceCorruptionError:
                        continue
                else:
                    raise RuntimeError("append never landed")
        except Exception as error:  # noqa: BLE001 - collected for the assert
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    _, events = handle.read()
    assert [event.seq for event in events] == list(range(10))
    path = tmp_path / "s-1" / "session.v4.jsonl"
    assert len(read_generation(path, recovery="recoverable").events) == 10


def test_handle_flush_skips_dir_fsync_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directory fsync is POSIX-only; a Windows flush still materializes."""
    _patch_windows(monkeypatch, _FakeMsvcrt())
    handle = _open_handle(tmp_path)
    handle.flush()
    path = tmp_path / "s-1" / "session.v4.jsonl"
    assert read_generation(path, recovery="recoverable").events == []


def test_handle_append_rejects_non_json_surface_metadata(tmp_path: Path) -> None:
    """Non-JSON surface metadata refuses the batch, log untouched."""
    handle, path = _materialized_handle(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="non-JSON-serializable surface metadata"):
        handle.append(
            [
                _session_event(
                    "s-1",
                    "turn/start",
                    3,
                    {"turn": 2},
                    surface_op={"op": object()},
                )
            ]
        )
    bad_seqs = SessionEvent.model_construct(
        id="s-1:3",
        session_id="s-1",
        type="turn/start",
        timestamp=1.003,
        payload={"turn": 2},
        ignorable=False,
        seq=3,
        surface_op=None,
        source_event_seqs=[object()],
    )
    with pytest.raises(ValueError, match="non-JSON-serializable surface metadata"):
        handle.append([bad_seqs])
    assert path.read_bytes() == before
    _, events = handle.read()
    assert len(events) == 3


def test_handle_append_skips_repair_when_unmaterialized(tmp_path: Path) -> None:
    """Torn state on a pending handle waits; the first append just writes."""
    handle = _open_handle(tmp_path, torn_truncate_to=5)
    handle.append([_session_event("s-1", "turn/start", 0, {"turn": 1})])
    path = tmp_path / "s-1" / "session.v4.jsonl"
    assert [event.seq for event in read_generation(path).events] == [0]


def test_handle_append_rewrites_recovered_tail_without_truncate(
    tmp_path: Path,
) -> None:
    """A recovered tail with no torn bytes is still rewritten before the batch."""
    header, rows = _good_turn()
    path = tmp_path / "s-1" / "session.v4.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_v4_log(path, header, rows[:2])
    result = read_generation(path, recovery="recoverable")
    recovered = _session_event(
        "s-1", "turn/end", 2, {"turn": 1, "reason": {"kind": "done"}}
    )
    handle = _open_handle(
        tmp_path,
        events=[*result.events, recovered],
        recovered_tail=[recovered],
        materialized=True,
        header_written=True,
    )
    handle.append([_session_event("s-1", "turn/start", 3, {"turn": 2})])
    reread = read_generation(path, recovery="recoverable")
    assert [event.seq for event in reread.events] == [0, 1, 2, 3]
    verify_current_generation(path)


class _FailingLease:
    """Stand-in lease whose release always fails."""

    def release(self) -> None:
        raise RuntimeError("lease boom")


def test_handle_close_combines_lease_and_claim_failures(tmp_path: Path) -> None:
    """Lease and claim failures on close combine into one group."""

    def failing_claim() -> None:
        raise RuntimeError("claim boom")

    handle, _path = _materialized_handle(
        tmp_path, lease=_FailingLease(), release_claim=failing_claim
    )
    with pytest.raises(ExceptionGroup) as excinfo:
        handle.close()
    assert handle.closed is True
    assert len(excinfo.value.exceptions) == 2
    assert any("lease boom" in str(error) for error in excinfo.value.exceptions)
    assert any("claim boom" in str(error) for error in excinfo.value.exceptions)
    handle.close()


def test_handle_append_keeps_valid_surface_metadata(tmp_path: Path) -> None:
    """A surface event keeps its operation and source seqs through the log."""
    handle, path = _materialized_handle(tmp_path)
    handle.append(
        [
            _session_event(
                "s-1",
                "user/message",
                3,
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                surface_op="append",
                source_event_seqs=[0, 1],
            )
        ]
    )
    reread = read_generation(path, recovery="recoverable")
    assert len(reread.events) == 4
    stored = reread.events[3]
    assert stored.type == "user/message"
    assert stored.surface_op == "append"
    assert stored.source_event_seqs == [0, 1]
    assert stored.payload["content"] == [{"type": "text", "text": "hi"}]


# ---------------------------------------------------------------------------
# Slice 2c: JsonlSessionPersistence service
# ---------------------------------------------------------------------------


def _persistence(tmp_path: Path) -> JsonlSessionPersistence:
    """One service rooted under the test's temp dir."""
    return JsonlSessionPersistence(tmp_path / "root")


def _write_current_log(root: Path, session_id: str = "s-1") -> Path:
    """Write a current-version log with the standard three-event turn."""
    path = root / "_no-cwd" / session_id / "session.v4.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    header, rows = _good_turn(session_id)
    _write_v4_log(path, header, rows)
    return path


def _write_historical_log(root: Path) -> Path:
    """Write a v0 log with the standard four-row turn."""
    path = root / "_no-cwd" / "s-1" / "session.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_v0_log(path, _v0_turn())
    return path


def test_service_create_returns_unmaterialized_handle(tmp_path: Path) -> None:
    """Create returns a write handle with no lease and no footprint."""
    svc = _persistence(tmp_path)
    handle = svc.create(_v4_header())
    assert handle.access == "write"
    assert handle.id == "s-1"
    assert handle.header["id"] == "s-1"
    assert handle.inherited_event_count == 0
    assert handle.closed is False
    state, events = handle.read()
    assert state == "detached"
    assert events == []
    assert not (tmp_path / "root").exists()
    handle.close()
    path = tmp_path / "root" / "_no-cwd" / "s-1" / "session.v4.jsonl"
    assert read_generation(path, recovery="recoverable").events == []


def test_service_create_pending_observable_via_stat_list_open(
    tmp_path: Path,
) -> None:
    """A created session is visible via stat/list/open before it materializes."""
    svc = _persistence(tmp_path)
    handle = svc.create(_v4_header())
    handle.append(_turn_events())
    snapshot = svc.stat("s-1")
    assert isinstance(snapshot, SessionPersistenceSnapshot)
    assert snapshot.format_status == "current"
    assert snapshot.revision == "pending:s-1"
    assert snapshot.event_count == 3
    assert snapshot.size_bytes is None
    assert snapshot.location is None
    assert snapshot.header["id"] == "s-1"
    listed = svc.list()
    assert [item.header["id"] for item in listed] == ["s-1"]
    reader = svc.open("s-1", "read")
    assert reader is not handle
    assert reader.access == "read"
    _, events = reader.read()
    assert [event.seq for event in events] == [0, 1, 2]
    # The read handle is a snapshot of the pending log at open time.
    handle.append([_session_event("s-1", "turn/start", 3, {"turn": 2})])
    _, events = reader.read()
    assert len(events) == 3
    with pytest.raises(SessionAlreadyOwnedError):
        svc.open("s-1", "write")
    reader.close()
    handle.close()


def test_service_create_twice_is_already_exists(tmp_path: Path) -> None:
    """A second create for a pending id is refused."""
    svc = _persistence(tmp_path)
    handle = svc.create(_v4_header())
    with pytest.raises(SessionAlreadyExistsError):
        svc.create(_v4_header())
    handle.close()


def test_service_create_existing_on_disk_is_already_exists(tmp_path: Path) -> None:
    """A create for an id with a stored generation is refused."""
    root = tmp_path / "root"
    _write_current_log(root)
    svc = _persistence(tmp_path)
    with pytest.raises(SessionAlreadyExistsError):
        svc.create(_v4_header())


def test_service_create_seeded_and_unseeded_inherited_count(tmp_path: Path) -> None:
    """Seeded headers need an explicit cut; unseeded ones need none or zero."""
    svc = _persistence(tmp_path)
    seeded = _v4_header(isSeeded=True)
    for bad in (None, -1, 1.5, True):
        with pytest.raises(ValueError):
            svc.create(seeded, bad)  # type: ignore[arg-type]
    handle = svc.create(seeded, 2)
    assert handle.inherited_event_count == 2
    handle.close()
    for bad in (1, -1):
        with pytest.raises(ValueError):
            svc.create(_v4_header("s-2"), bad)
    handle2 = svc.create(_v4_header("s-2"), 0)
    assert handle2.inherited_event_count == 0
    handle2.close()
    handle3 = svc.create(_v4_header("s-3"))
    assert handle3.inherited_event_count == 0
    handle3.close()


def test_service_create_invalid_header_and_cwd(tmp_path: Path) -> None:
    """A header that fails validation, or a non-string cwd, is refused."""
    svc = _persistence(tmp_path)
    with pytest.raises(SessionFormatError):
        svc.create(
            {
                "version": 4,
                "createdAt": 1000,
                "isSeeded": False,
                "delegationDepth": 0,
            }
        )
    with pytest.raises(ValueError, match="cwd must be a string"):
        svc.create(_v4_header(cwd=123))


def test_service_create_with_cwd_uses_project_dir(tmp_path: Path) -> None:
    """A header cwd selects the project dir; open finds it without the cwd."""
    svc = _persistence(tmp_path)
    handle = svc.create(_v4_header(cwd="/work/proj"))
    handle.append(_turn_events())
    path = tmp_path / "root" / "--work-proj--" / "s-1" / "session.v4.jsonl"
    assert path.is_file()
    handle.close()
    snapshot = svc.stat("s-1")
    assert snapshot is not None
    assert snapshot.location == path
    reopened = svc.open("s-1", "write")
    assert reopened.header["cwd"] == "/work/proj"
    _, events = reopened.read()
    assert len(events) == 3
    reopened.close()


def test_service_open_missing_is_not_found(tmp_path: Path) -> None:
    """Open for either access on an unknown id is NotFound."""
    svc = _persistence(tmp_path)
    with pytest.raises(SessionPersistenceNotFoundError):
        svc.open("missing", "read")
    with pytest.raises(SessionPersistenceNotFoundError):
        svc.open("missing", "write")


def test_service_open_read_on_disk_current(tmp_path: Path) -> None:
    """A read open serves the stored log as shared-frozen."""
    root = tmp_path / "root"
    _write_current_log(root)
    svc = _persistence(tmp_path)
    handle = svc.open("s-1", "read")
    assert handle.access == "read"
    state, events = handle.read()
    assert state == "shared-frozen"
    assert [event.type for event in events] == [
        "turn/start",
        "step/start",
        "turn/end",
    ]
    handle.close()


def test_service_open_read_historical_primed_without_publication(
    tmp_path: Path,
) -> None:
    """A read open primes migrated events and publishes nothing."""
    root = tmp_path / "root"
    source = _write_historical_log(root)
    target = root / "_no-cwd" / "s-1" / "session.v4.jsonl"
    svc = _persistence(tmp_path)
    handle = svc.open("s-1", "read")
    _, events = handle.read()
    assert [event.type for event in events] == [
        "turn/start",
        "step/start",
        "system/message",
        "step/end",
        "turn/end",
    ]
    assert source.exists()
    assert not target.exists()
    handle.close()
    assert source.exists()
    assert not target.exists()


def test_service_open_read_header_id_mismatch_is_corruption(tmp_path: Path) -> None:
    """A stored header naming another id is corruption on read-open."""
    root = tmp_path / "root"
    path = root / "_no-cwd" / "s-1" / "session.v4.jsonl"
    path.parent.mkdir(parents=True)
    header, rows = _good_turn("other")
    _write_v4_log(path, header, rows)
    svc = _persistence(tmp_path)
    with pytest.raises(SessionPersistenceCorruptionError, match="does not match"):
        svc.open("s-1", "read")


def test_service_open_write_ownership_and_release(tmp_path: Path) -> None:
    """A second write-open is owned; read works while write is held; close frees it."""
    root = tmp_path / "root"
    _write_current_log(root)
    svc = _persistence(tmp_path)
    first = svc.open("s-1", "write")
    with pytest.raises(SessionAlreadyOwnedError):
        svc.open("s-1", "write")
    reader = svc.open("s-1", "read")
    _, events = reader.read()
    assert len(events) == 3
    first.close()
    second = svc.open("s-1", "write")
    assert second.access == "write"
    second.close()
    reader.close()


def test_service_open_write_current_log_appends_after_cursor(tmp_path: Path) -> None:
    """A write-open handle continues the stored log at its end."""
    root = tmp_path / "root"
    path = _write_current_log(root)
    svc = _persistence(tmp_path)
    handle = svc.open("s-1", "write")
    handle.append([_session_event("s-1", "turn/start", 3, {"turn": 2})])
    assert len(read_generation(path, recovery="recoverable").events) == 4
    handle.close()


def test_service_open_write_header_id_mismatch_releases_claim(
    tmp_path: Path,
) -> None:
    """A failed write-open releases its claim: fixing the log lets the next one in."""
    root = tmp_path / "root"
    path = root / "_no-cwd" / "s-1" / "session.v4.jsonl"
    path.parent.mkdir(parents=True)
    header, rows = _good_turn("other")
    _write_v4_log(path, header, rows)
    svc = _persistence(tmp_path)
    with pytest.raises(SessionPersistenceCorruptionError, match="does not match"):
        svc.open("s-1", "write")
    header, rows = _good_turn("s-1")
    _write_v4_log(path, header, rows)
    handle = svc.open("s-1", "write")
    _, events = handle.read()
    assert len(events) == 3
    handle.close()


def test_service_open_write_kernel_lease_contention_across_instances(
    tmp_path: Path,
) -> None:
    """A second service instance contends on the kernel lease, not the claim."""
    root = tmp_path / "root"
    _write_current_log(root)
    first_svc = _persistence(tmp_path)
    second_svc = _persistence(tmp_path)
    handle = first_svc.open("s-1", "write")
    with pytest.raises(SessionAlreadyOwnedError):
        second_svc.open("s-1", "write")
    reader = second_svc.open("s-1", "read")
    _, events = reader.read()
    assert len(events) == 3
    handle.close()
    reopened = second_svc.open("s-1", "write")
    assert reopened.access == "write"
    reopened.close()
    reader.close()


def test_service_create_then_append_takes_lease_across_instances(
    tmp_path: Path,
) -> None:
    """A created handle takes the kernel lease on its first materializing
    write, so a second instance cannot write-open it until it closes."""
    first_svc = _persistence(tmp_path)
    second_svc = _persistence(tmp_path)
    handle = first_svc.create(_v4_header("s-1"))
    handle.append([_session_event("s-1", "turn/start", 0, {"turn": 1})])
    with pytest.raises(SessionAlreadyOwnedError):
        second_svc.open("s-1", "write")
    handle.close()
    reopened = second_svc.open("s-1", "write")
    _, events = reopened.read()
    assert len(events) == 1
    reopened.close()


def test_service_open_unknown_access_is_value_error(tmp_path: Path) -> None:
    """An access value outside read/write is refused."""
    svc = _persistence(tmp_path)
    with pytest.raises(ValueError, match="unknown session access"):
        svc.open("s-1", "peek")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Slice 2c: stat / list
# ---------------------------------------------------------------------------


def test_service_stat_on_disk_current_and_missing(tmp_path: Path) -> None:
    """Stat reports header-only facts with the file-identity revision."""
    root = tmp_path / "root"
    path = _write_current_log(root)
    svc = _persistence(tmp_path)
    snapshot = svc.stat("s-1")
    assert snapshot is not None
    assert snapshot.format_status == "current"
    assert snapshot.header["id"] == "s-1"
    assert snapshot.event_count is None
    assert snapshot.size_bytes == path.stat().st_size
    assert snapshot.location == path
    info = path.stat()
    assert snapshot.revision == (
        f"{info.st_dev}:{info.st_ino}:{info.st_size}:"
        f"{info.st_mtime_ns}:{info.st_ctime_ns}"
    )
    assert svc.stat("missing") is None


def test_service_stat_pending_becomes_on_disk_after_close(tmp_path: Path) -> None:
    """A pending snapshot turns into a file snapshot once the handle closes."""
    svc = _persistence(tmp_path)
    handle = svc.create(_v4_header())
    handle.append(_turn_events())
    pending = svc.stat("s-1")
    assert pending is not None
    assert pending.revision == "pending:s-1"
    assert pending.event_count == 3
    handle.close()
    snapshot = svc.stat("s-1")
    assert snapshot is not None
    assert snapshot.format_status == "current"
    assert snapshot.revision != "pending:s-1"
    assert snapshot.event_count is None
    assert snapshot.location is not None
    assert snapshot.location.is_file()


def test_service_stat_pending_closed_mid_read_falls_back_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending handle that closes mid-stat falls back to the stored log."""
    svc = _persistence(tmp_path)
    handle = svc.create(_v4_header())
    handle.append(_turn_events())
    handle.flush()
    original_read = handle.read

    def closing_read(*args: object, **kwargs: object) -> object:
        raise SessionHandleClosedError("s-1", "read")

    monkeypatch.setattr(handle, "read", closing_read)
    snapshot = svc.stat("s-1")
    assert snapshot is not None
    assert snapshot.revision != "pending:s-1"
    assert snapshot.location is not None
    assert snapshot.format_status == "current"
    monkeypatch.undo()
    assert handle.read == original_read
    handle.close()


def test_service_stat_historical_is_migration_required(tmp_path: Path) -> None:
    """Stat classifies a historical generation without reading its rows."""
    root = tmp_path / "root"
    source = _write_historical_log(root)
    svc = _persistence(tmp_path)
    snapshot = svc.stat("s-1")
    assert snapshot is not None
    assert snapshot.format_status == "migration-required"
    assert snapshot.event_count is None
    assert snapshot.location == source


def test_service_stat_newer_version_has_no_status_and_does_not_raise(
    tmp_path: Path,
) -> None:
    """A newer-version header stats with no format status, never a raise."""
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    session_path.mkdir(parents=True)
    log = session_path / "session.v5.jsonl"
    log.write_text(
        _dump(
            {
                "type": "session",
                "version": 5,
                "id": "s-1",
                "createdAt": 1000,
                "isSeeded": False,
                "delegationDepth": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    svc = _persistence(tmp_path)
    snapshot = svc.stat("s-1")
    assert snapshot is not None
    assert snapshot.format_status is None
    assert snapshot.header["id"] == "s-1"
    assert snapshot.header["version"] == 5
    assert snapshot.location == log
    assert snapshot.size_bytes == log.stat().st_size


def test_service_stat_malformed_header_is_corruption(tmp_path: Path) -> None:
    """A malformed stored header is damage, not a version issue."""
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    session_path.mkdir(parents=True)
    log = session_path / "session.v4.jsonl"
    log.write_text(
        _dump(
            {
                "type": "session",
                "version": 4,
                "id": 3,
                "createdAt": 1000,
                "isSeeded": False,
                "delegationDepth": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    svc = _persistence(tmp_path)
    with pytest.raises(SessionPersistenceCorruptionError):
        svc.stat("s-1")


def test_service_stat_unreadable_first_line_is_corruption(tmp_path: Path) -> None:
    """An empty log and an unparsable first line have no header to classify."""
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    session_path.mkdir(parents=True)
    empty = session_path / "session.v4.jsonl"
    empty.write_bytes(b"")
    svc = _persistence(tmp_path)
    with pytest.raises(
        SessionPersistenceCorruptionError, match="first line is not a session header"
    ):
        svc.stat("s-1")
    empty.write_bytes(b"{bad json\n")
    with pytest.raises(
        SessionPersistenceCorruptionError, match="corrupt session header"
    ):
        svc.stat("s-1")


def test_service_stat_header_id_mismatch_is_corruption(tmp_path: Path) -> None:
    """Stat refuses a stored header naming another id."""
    root = tmp_path / "root"
    path = root / "_no-cwd" / "s-1" / "session.v4.jsonl"
    path.parent.mkdir(parents=True)
    header, rows = _good_turn("other")
    _write_v4_log(path, header, rows)
    svc = _persistence(tmp_path)
    with pytest.raises(SessionPersistenceCorruptionError, match="does not match"):
        svc.stat("s-1")


def test_service_list_covers_pending_and_on_disk(tmp_path: Path) -> None:
    """List merges pending entries with every stored generation, once each."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    _write_current_log(root, "s-1")
    historical_dir = root / "--work-proj--" / "s-2"
    historical_dir.mkdir(parents=True)
    _write_v0_log(historical_dir / "session.jsonl", _v0_turn(), "s-2")
    (root / "_no-cwd" / "s-3").mkdir(parents=True)
    (root / "stray.txt").write_text("not a project", encoding="utf-8")
    (root / "_no-cwd" / "stray.txt").write_text("not a session", encoding="utf-8")
    pending = svc.create(_v4_header("s-4"))
    pending.append([_session_event("s-4", "turn/start", 0, {"turn": 1})])
    listed = svc.list()
    by_id = {item.header["id"]: item for item in listed}
    assert set(by_id) == {"s-1", "s-2", "s-4"}
    assert by_id["s-1"].format_status == "current"
    assert by_id["s-2"].format_status == "migration-required"
    assert by_id["s-4"].format_status == "current"
    assert by_id["s-4"].revision == "pending:s-4"
    assert by_id["s-4"].event_count == 1
    pending.close()


def test_service_list_empty_root(tmp_path: Path) -> None:
    """List on a missing root is empty, not an error."""
    assert _persistence(tmp_path).list() == []


def test_service_open_read_pending_closed_mid_read_falls_back_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending handle that closes mid-open falls back to the stored log."""
    svc = _persistence(tmp_path)
    handle = svc.create(_v4_header())
    handle.append(_turn_events())
    handle.flush()

    def closing_read(*args: object, **kwargs: object) -> object:
        raise SessionHandleClosedError("s-1", "read")

    monkeypatch.setattr(handle, "read", closing_read)
    reader = svc.open("s-1", "read")
    _, events = reader.read()
    assert len(events) == 3
    monkeypatch.undo()
    reader.close()
    handle.close()


def test_service_list_skips_pending_handle_closed_mid_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pending handle that closes mid-list is skipped, not fatal."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    _write_current_log(root, "s-1")
    closing = svc.create(_v4_header("s-2"))

    def closing_read(*args: object, **kwargs: object) -> object:
        raise SessionHandleClosedError("s-2", "read")

    monkeypatch.setattr(closing, "read", closing_read)
    listed = svc.list()
    assert [item.header["id"] for item in listed] == ["s-1"]
    monkeypatch.undo()
    closing.close()


# ---------------------------------------------------------------------------
# Slice 2c: flush
# ---------------------------------------------------------------------------


def test_service_flush_with_no_handles_is_noop(tmp_path: Path) -> None:
    """Flushing a service with no active handles succeeds silently."""
    _persistence(tmp_path).flush()


def test_service_flush_flushes_all_owned_handles(tmp_path: Path) -> None:
    """A service flush materializes every owned handle, pending included."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    first = svc.create(_v4_header("s-1"))
    second = svc.create(_v4_header("s-2"))
    first.append([_session_event("s-1", "turn/start", 0, {"turn": 1})])
    svc.flush()
    first_path = root / "_no-cwd" / "s-1" / "session.v4.jsonl"
    second_path = root / "_no-cwd" / "s-2" / "session.v4.jsonl"
    assert first_path.is_file()
    assert second_path.is_file()
    assert len(read_generation(second_path, recovery="recoverable").events) == 0
    first.close()
    second.close()


def test_service_flush_groups_failures_and_flushes_the_rest(
    tmp_path: Path,
) -> None:
    """One failing handle groups its failure by id; the rest still flush."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    blocker = root / "_no-cwd" / "s-1"
    blocker.parent.mkdir(parents=True)
    blocker.write_text("not a directory", encoding="utf-8")
    failing = svc.create(_v4_header("s-1"))
    good = svc.create(_v4_header("s-2"))
    with pytest.raises(ExceptionGroup) as excinfo:
        svc.flush()
    assert "s-1" in str(excinfo.value)
    assert len(excinfo.value.exceptions) == 1
    assert isinstance(excinfo.value.exceptions[0], RuntimeError)
    assert "s-1" in str(excinfo.value.exceptions[0])
    assert (root / "_no-cwd" / "s-2" / "session.v4.jsonl").is_file()
    good.close()
    with pytest.raises(ExceptionGroup):
        failing.close()


def test_service_flush_skips_concurrently_closed_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handles closed mid-flush count as flushed; the rest still flush."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    first = svc.create(_v4_header("s-1"))
    second = svc.create(_v4_header("s-2"))
    monkeypatch.setattr(JsonlSessionHandle, "closed", property(lambda self: True))
    svc.flush()
    assert not (root / "_no-cwd" / "s-1" / "session.v4.jsonl").exists()
    assert not (root / "_no-cwd" / "s-2" / "session.v4.jsonl").exists()
    monkeypatch.undo()

    def raising_flush() -> None:
        raise SessionHandleClosedError("s-1", "flush")

    monkeypatch.setattr(first, "flush", raising_flush)
    svc.flush()
    assert (root / "_no-cwd" / "s-2" / "session.v4.jsonl").is_file()
    monkeypatch.undo()
    first.close()
    second.close()


# ---------------------------------------------------------------------------
# Slice 2c: migration publication
# ---------------------------------------------------------------------------


def test_service_open_write_publishes_historical_migration(tmp_path: Path) -> None:
    """Write-open publishes a historical log: v4 present, source gone, events intact."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    handle = svc.open("s-1", "write")
    assert target.is_file()
    assert not source.exists()
    assert not list(session_path.glob("*.tmp-*"))
    _, events = handle.read()
    assert [event.type for event in events] == [
        "turn/start",
        "step/start",
        "system/message",
        "step/end",
        "turn/end",
    ]
    assert [event.seq for event in events] == [0, 1, 2, 3, 4]
    # Millisecond times survive exactly; 1.001 is the value truncation
    # would have drifted down by a millisecond.
    assert events[0].timestamp == 1.0
    assert events[1].timestamp == 1.001
    assert events[3].timestamp == 1.002
    assert events[4].timestamp == 1.003
    reread = read_generation(target, recovery="recoverable")
    assert [event.timestamp for event in reread.events] == [
        event.timestamp for event in events
    ]
    assert reread.format_status == "current"
    handle.append([_session_event("s-1", "turn/start", 5, {"turn": 2})])
    assert len(read_generation(target, recovery="recoverable").events) == 6
    verify_current_generation(target)
    handle.close()
    reopened = svc.open("s-1", "write")
    _, events = reopened.read()
    assert len(events) == 6
    reopened.close()


def _v0_turns(count: int) -> list[dict]:
    """N consecutive v0 turns with distinct millisecond times."""
    rows: list[dict] = []
    seq = 0
    for turn in range(1, count + 1):
        rows.append(
            _v4_event(
                "turn/start",
                seq,
                1000 + seq,
                {"turn": turn, "trigger": {"kind": "user"}},
            )
        )
        seq += 1
        rows.append(_v4_event("step/start", seq, 1000 + seq, {"turn": turn, "step": 1}))
        seq += 1
        rows.append(_v4_event("step/end", seq, 1000 + seq, {"turn": turn, "step": 1}))
        seq += 1
        rows.append(
            _v4_event(
                "turn/end",
                seq,
                1000 + seq,
                {"turn": turn, "reason": {"kind": "done"}},
            )
        )
        seq += 1
    return rows


def test_service_open_write_publishes_long_historical_migration(
    tmp_path: Path,
) -> None:
    """The prefix digest covers only the first sixteen of a longer log."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    session_path.mkdir(parents=True)
    source = _write_v0_log(session_path / "session.jsonl", _v0_turns(5))
    handle = svc.open("s-1", "write")
    assert not source.exists()
    _, events = handle.read()
    assert len(events) == 21
    assert [event.seq for event in events] == list(range(21))
    handle.append([_session_event("s-1", "turn/start", 21, {"turn": 6})])
    assert len(read_generation(session_path / "session.v4.jsonl").events) == 22
    handle.close()


def _patched_verify(
    monkeypatch: pytest.MonkeyPatch, fake: ReadResult | BaseException
) -> None:
    """Patch the module's verification read to return a fake result or raise."""

    def verify(path: Path) -> ReadResult:
        if isinstance(fake, BaseException):
            raise fake
        return fake

    monkeypatch.setattr(
        "magitech_harness.session.persistence_jsonl.verify_current_generation",
        verify,
    )


def test_service_open_write_verification_failure_keeps_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing verification removes the new file and keeps the source."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    _patched_verify(
        monkeypatch,
        SessionPersistenceCorruptionError("verification boom"),
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="boom"):
        svc.open("s-1", "write")
    assert source.exists()
    assert not target.exists()
    assert not list(session_path.glob("*.tmp-*"))
    monkeypatch.undo()
    handle = svc.open("s-1", "write")
    assert target.is_file()
    assert not source.exists()
    handle.close()


def test_service_open_write_verification_id_mismatch_keeps_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified artifact naming another id is rejected; the source stays."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    stored = read_generation(source, recovery="recoverable")
    fake = dataclasses.replace(stored, header={**stored.header, "id": "other"})
    _patched_verify(monkeypatch, fake)
    with pytest.raises(SessionPersistenceCorruptionError, match="failed verification"):
        svc.open("s-1", "write")
    assert source.exists()
    assert not target.exists()


def test_service_open_write_verification_count_mismatch_keeps_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified artifact with the wrong event count is rejected."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    stored = read_generation(source, recovery="recoverable")
    fake = dataclasses.replace(stored, events=stored.events[:-1])
    _patched_verify(monkeypatch, fake)
    with pytest.raises(SessionPersistenceCorruptionError, match="failed verification"):
        svc.open("s-1", "write")
    assert source.exists()
    assert not target.exists()


def test_service_open_write_verification_digest_mismatch_keeps_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified artifact with a changed prefix is rejected by its digest."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    stored = read_generation(source, recovery="recoverable")
    changed = stored.events[0].model_copy(update={"payload": {"turn": 99}})
    fake = dataclasses.replace(stored, events=[changed, *stored.events[1:]])
    _patched_verify(monkeypatch, fake)
    with pytest.raises(
        SessionPersistenceCorruptionError, match="prefix digest mismatch"
    ):
        svc.open("s-1", "write")
    assert source.exists()
    assert not target.exists()


def test_service_open_write_verification_cleanup_failure_still_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cleanup failure while removing a rejected artifact keeps the error."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"

    def failing_verify(path: Path) -> ReadResult:
        path.unlink()
        path.mkdir()
        raise SessionPersistenceCorruptionError("boom")

    monkeypatch.setattr(
        "magitech_harness.session.persistence_jsonl.verify_current_generation",
        failing_verify,
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="boom"):
        svc.open("s-1", "write")
    assert source.exists()
    assert target.is_dir()


def test_service_open_write_stale_temp_file_is_replaced(tmp_path: Path) -> None:
    """A stale temp artifact from an earlier run does not block publication."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    tmp_path = session_path / f"session.v4.jsonl.tmp-{os.getpid()}"
    tmp_path.write_text("stale garbage", encoding="utf-8")
    handle = svc.open("s-1", "write")
    assert target.is_file()
    assert not source.exists()
    assert not list(session_path.glob("*.tmp-*"))
    handle.close()


def test_service_open_write_temp_path_blocked_keeps_source(tmp_path: Path) -> None:
    """A temp path blocked by a directory fails loud; the source stays."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    tmp_dir = session_path / f"session.v4.jsonl.tmp-{os.getpid()}"
    tmp_dir.mkdir()
    # The errno differs by platform (a directory may surface as a file
    # existing, a directory, or a denied write); the contract is only
    # that publication fails loud and touches nothing it did not create.
    with pytest.raises(OSError):
        svc.open("s-1", "write")
    assert source.exists()
    assert not target.exists()
    assert tmp_dir.is_dir()
    tmp_dir.rmdir()
    handle = svc.open("s-1", "write")
    assert target.is_file()
    assert not source.exists()
    handle.close()


def test_service_open_write_temp_path_symlink_keeps_source(tmp_path: Path) -> None:
    """A temp path blocked by a symlink fails loud; the link stays."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    blocker = session_path / "blocker"
    blocker.mkdir()
    tmp_link = session_path / f"session.v4.jsonl.tmp-{os.getpid()}"
    os.symlink(blocker, tmp_link, target_is_directory=True)
    with pytest.raises(OSError):
        svc.open("s-1", "write")
    assert source.exists()
    assert not target.exists()
    assert tmp_link.is_symlink()
    tmp_link.unlink()
    handle = svc.open("s-1", "write")
    assert target.is_file()
    assert not source.exists()
    handle.close()


def test_service_open_write_target_blocked_cleans_temp_keeps_source(
    tmp_path: Path,
) -> None:
    """A blocked target fails publication; the temp file is removed."""
    svc = _persistence(tmp_path)
    root = tmp_path / "root"
    session_path = root / "_no-cwd" / "s-1"
    source = _write_historical_log(root)
    target = session_path / "session.v4.jsonl"
    target.mkdir()
    with pytest.raises(OSError):
        svc.open("s-1", "write")
    assert source.exists()
    assert target.is_dir()
    assert not list(session_path.glob("*.tmp-*"))
    target.rmdir()
    handle = svc.open("s-1", "write")
    assert target.is_file()
    assert not source.exists()
    handle.close()


# ---------------------------------------------------------------------------
# Slice 2c: migration event rows
# ---------------------------------------------------------------------------


def test_migration_event_row_round_trips_time_and_surface() -> None:
    """A restored event keeps its millisecond time and surface metadata."""
    event = _session_event(
        "s-1",
        "user/message",
        3,
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        surface_op="append",
        source_event_seqs=[0, 1],
    )
    row = _migration_event_row(event, "s-1")
    assert row["type"] == "user/message"
    assert row["seq"] == 3
    assert row["time"] == 1003
    assert row["surfaceOp"] == "append"
    assert row["sourceEventSeqs"] == [0, 1]
    assert "ignorable" not in row


def test_migration_event_row_marks_ignorable_events() -> None:
    """An ignorable event keeps its marker; a plain event carries none."""
    ignorable = _session_event("s-1", "turn/start", 0, {"turn": 1}, ignorable=True)
    assert _migration_event_row(ignorable, "s-1")["ignorable"] is True
    plain = _session_event("s-1", "turn/start", 0, {"turn": 1})
    assert "ignorable" not in _migration_event_row(plain, "s-1")


def test_migration_event_row_rejects_retired_type() -> None:
    """An event that fails current admission cannot be published."""
    event = _session_event("s-1", "tool/code-dispatch-start", 0, {"turn": 1})
    with pytest.raises(ValueError, match="fails v4 admission"):
        _migration_event_row(event, "s-1")


# ---------------------------------------------------------------------------
# Live tracker
# ---------------------------------------------------------------------------


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> bool:
    """Poll a predicate until true or the deadline passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _tracker_setup(
    tmp_path: Path, *, session_id: str = "s-1", delay: float = 0.01
) -> tuple[JsonlSessionPersistence, JsonlSessionHandle, JsonlLiveTracker, Path]:
    """A persistence with one created session and a tracker over it."""
    persistence = JsonlSessionPersistence(tmp_path / "root", live_batch_delay=delay)
    handle = persistence.create(_v4_header(session_id))
    tracker = JsonlLiveTracker(persistence)
    path = tmp_path / "root" / "_no-cwd" / session_id / "session.v4.jsonl"
    return persistence, handle, tracker, path


def _stop_tracker(tracker: JsonlLiveTracker) -> None:
    """Stop a tracker that was never installed: cancel pending timers."""
    dispose = tracker.install(Context())
    dispose()


def _fake_session(session_id: str) -> types.SimpleNamespace:
    """A minimal session stand-in carrying only its id."""
    return types.SimpleNamespace(session_id=session_id)


def _stored_seqs(path: Path) -> list[int]:
    """Seqs in the durable log, or an empty list when nothing landed."""
    if not path.exists():
        return []
    return [event.seq for event in read_generation(path).events]


def test_tracker_enqueue_buffers_without_writing(tmp_path: Path) -> None:
    """Enqueued events stay in memory until a drain or the timer."""
    _persistence, handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    try:
        tracker.enqueue_live(
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 0, {"turn": 1}),
        )
        assert not path.exists()
        _, events = handle.read()
        assert events == []
    finally:
        _stop_tracker(tracker)


def test_tracker_timer_fires_and_drains(tmp_path: Path) -> None:
    """The batching timer drains the buffer durably, in order."""
    _persistence, handle, tracker, path = _tracker_setup(tmp_path)
    for event in _turn_events():
        tracker.enqueue_live(_fake_session("s-1"), event)
    assert _wait_until(lambda: _stored_seqs(path) == [0, 1, 2])
    _, events = handle.read()
    assert [event.payload for event in events] == [
        {"turn": 1},
        {"turn": 1, "step": 1},
        {"turn": 1, "reason": {"kind": "done"}},
    ]


def test_tracker_explicit_drain_writes_batch_durably(tmp_path: Path) -> None:
    """An explicit drain writes the batch; a second drain is a no-op."""
    _persistence, handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    try:
        for event in _turn_events():
            tracker.enqueue_live(_fake_session("s-1"), event)
        assert not path.exists()
        tracker.drain_live("s-1")
        assert _stored_seqs(path) == [0, 1, 2]
        tracker.drain_live("s-1")
        tracker.drain_live()
        assert _stored_seqs(path) == [0, 1, 2]
        _, events = handle.read()
        assert len(events) == 3
    finally:
        _stop_tracker(tracker)


def test_tracker_drain_all_writes_every_session(tmp_path: Path) -> None:
    """A drain with no id writes every buffered session."""
    persistence, _handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    try:
        other = persistence.create(_v4_header("s-2"))
        tracker.enqueue_live(
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 0, {"turn": 1}),
        )
        tracker.enqueue_live(
            _fake_session("s-2"),
            _session_event("s-2", "turn/start", 0, {"turn": 1}),
        )
        tracker.drain_live()
        assert _stored_seqs(path) == [0]
        other_path = tmp_path / "root" / "_no-cwd" / "s-2" / "session.v4.jsonl"
        assert _stored_seqs(other_path) == [0]
        assert other.closed is False
    finally:
        _stop_tracker(tracker)


def test_tracker_enqueue_clones_and_rejects_non_json(tmp_path: Path) -> None:
    """The buffered copy is detached; non-JSON values refuse loudly."""
    _persistence, _handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    try:
        event = _session_event("s-1", "turn/start", 0, {"turn": 1})
        tracker.enqueue_live(_fake_session("s-1"), event)
        event.payload["turn"] = 99
        tracker.drain_live("s-1")
        stored = read_generation(path).events
        assert [item.payload for item in stored] == [{"turn": 1}]
        bad_payload = _session_event(
            "s-1", "turn/start", 1, {"turn": 2, "bad": object()}
        )
        with pytest.raises(ValueError, match="non-JSON-serializable data"):
            tracker.enqueue_live(_fake_session("s-1"), bad_payload)
        bad_surface = _session_event(
            "s-1", "turn/start", 1, {"turn": 2}, surface_op={"op": object()}
        )
        with pytest.raises(ValueError, match="non-JSON-serializable surface metadata"):
            tracker.enqueue_live(_fake_session("s-1"), bad_surface)
        bad_seqs = SessionEvent.model_construct(
            id="s-1:1",
            session_id="s-1",
            type="turn/start",
            timestamp=1.001,
            payload={"turn": 2},
            ignorable=False,
            seq=1,
            surface_op=None,
            source_event_seqs=[object()],
        )
        with pytest.raises(ValueError, match="non-JSON-serializable surface metadata"):
            tracker.enqueue_live(_fake_session("s-1"), bad_seqs)
    finally:
        _stop_tracker(tracker)


def test_tracker_enqueue_clones_surface_metadata(tmp_path: Path) -> None:
    """A cloned event keeps valid surface metadata through the drain."""
    _persistence, handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    try:
        handle.append(
            [
                _session_event("s-1", "turn/start", 0, {"turn": 1}),
                _session_event("s-1", "step/start", 1, {"turn": 1, "step": 1}),
            ]
        )
        tracker.enqueue_live(
            _fake_session("s-1"),
            _session_event(
                "s-1",
                "user/message",
                2,
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                surface_op="append",
                source_event_seqs=[0, 1],
            ),
        )
        tracker.drain_live("s-1")
        stored = read_generation(path).events
        assert stored[2].surface_op == "append"
        assert stored[2].source_event_seqs == [0, 1]
    finally:
        _stop_tracker(tracker)


def test_tracker_events_without_write_handle_are_ignored(tmp_path: Path) -> None:
    """Events for an unclaimed session are dropped silently."""
    persistence = JsonlSessionPersistence(tmp_path / "root", live_batch_delay=0.01)
    tracker = JsonlLiveTracker(persistence)
    tracker.enqueue_live(
        _fake_session("s-1"),
        _session_event("s-1", "turn/start", 0, {"turn": 1}),
    )
    tracker.enqueue_live("s-1", _session_event("s-1", "turn/start", 0, {"turn": 1}))
    tracker.drain_live("s-1")
    tracker.drain_live()
    assert not (tmp_path / "root").exists()


def test_tracker_drain_failure_keeps_batch_and_pauses_timer(
    tmp_path: Path,
) -> None:
    """A failed drain keeps the batch, pauses the timer, retries loudly."""
    _persistence, handle, tracker, path = _tracker_setup(tmp_path)
    tracker.enqueue_live(
        _fake_session("s-1"), _session_event("s-1", "turn/start", 5, {"turn": 1})
    )
    with pytest.raises(SessionPersistenceCorruptionError, match="not contiguous"):
        tracker.drain_live("s-1")
    assert not path.exists()
    # A new event while paused starts no timer: nothing lands on its own.
    tracker.enqueue_live(
        _fake_session("s-1"), _session_event("s-1", "turn/start", 6, {"turn": 2})
    )
    time.sleep(0.15)
    assert not path.exists()
    with pytest.raises(SessionPersistenceCorruptionError, match="not contiguous"):
        tracker.drain_live("s-1")
    # Fill the gap directly, then the retry lands both buffered events.
    handle.append(
        [
            _session_event("s-1", "turn/start", 0, {"turn": 1}),
            _session_event("s-1", "step/start", 1, {"turn": 1, "step": 1}),
            _session_event(
                "s-1", "turn/end", 2, {"turn": 1, "reason": {"kind": "done"}}
            ),
            _session_event("s-1", "turn/start", 3, {"turn": 2}),
            _session_event(
                "s-1", "turn/end", 4, {"turn": 2, "reason": {"kind": "done"}}
            ),
        ]
    )
    tracker.drain_live("s-1")
    assert _stored_seqs(path) == [0, 1, 2, 3, 4, 5, 6]


def test_tracker_timer_failure_warns_and_pauses(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A timer-thread drain failure warns, keeps the batch, pauses."""
    _persistence, handle, tracker, path = _tracker_setup(tmp_path)
    tracker.enqueue_live(
        _fake_session("s-1"), _session_event("s-1", "turn/start", 5, {"turn": 1})
    )
    with caplog.at_level(logging.WARNING, logger="harness.session"):
        assert _wait_until(
            lambda: any(
                "live drain failed" in record.getMessage() for record in caplog.records
            )
        )
    assert not path.exists()
    with pytest.raises(SessionPersistenceCorruptionError, match="not contiguous"):
        tracker.drain_live("s-1")
    handle.append(
        [
            _session_event("s-1", "turn/start", 0, {"turn": 1}),
            _session_event("s-1", "step/start", 1, {"turn": 1, "step": 1}),
            _session_event(
                "s-1", "turn/end", 2, {"turn": 1, "reason": {"kind": "done"}}
            ),
            _session_event("s-1", "turn/start", 3, {"turn": 2}),
            _session_event(
                "s-1", "turn/end", 4, {"turn": 2, "reason": {"kind": "done"}}
            ),
        ]
    )
    tracker.drain_live("s-1")
    assert _stored_seqs(path) == [0, 1, 2, 3, 4, 5]


def test_tracker_drain_without_handle_keeps_batch(tmp_path: Path) -> None:
    """A closed handle leaves the batch buffered until a reopen drains it."""
    persistence, handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    try:
        tracker.enqueue_live(
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 0, {"turn": 1}),
        )
        handle.close()
        # Closing materialized the header; the buffered event is unwritten.
        assert _stored_seqs(path) == []
        tracker.drain_live("s-1")
        assert _stored_seqs(path) == []
        reopened = persistence.open("s-1", "write")
        tracker.drain_live("s-1")
        assert _stored_seqs(path) == [0]
        reopened.close()
    finally:
        _stop_tracker(tracker)


def test_tracker_flush_listener_drains_then_flushes(tmp_path: Path) -> None:
    """A flush event drains the buffer, then flushes the write handle."""
    _persistence, handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    ctx = Context()
    dispose = tracker.install(ctx)
    try:
        # No events yet: flush alone materializes the header.
        ctx.emit("session/flush", _fake_session("s-1"))
        assert path.exists()
        assert read_generation(path).events == []
        ctx.emit(
            "session/event",
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 0, {"turn": 1}),
        )
        assert not _stored_seqs(path)
        ctx.emit("session/flush", _fake_session("s-1"))
        assert _stored_seqs(path) == [0]
        _, events = handle.read()
        assert len(events) == 1
        # A session with no write handle flushes silently.
        ctx.emit("session/flush", _fake_session("ghost"))
    finally:
        dispose()


def test_tracker_disposed_final_drain_warns_instead_of_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failing final drain on disposal warns; the batch stays buffered."""
    _persistence, _handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    ctx = Context()
    dispose = tracker.install(ctx)
    try:
        tracker.enqueue_live(
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 5, {"turn": 1}),
        )
        with caplog.at_level(logging.WARNING, logger="harness.session"):
            ctx.emit("session/disposed", _fake_session("s-1"))
        assert any(
            "final live drain failed" in record.getMessage()
            for record in caplog.records
        )
        assert not path.exists()
        with pytest.raises(SessionPersistenceCorruptionError, match="not contiguous"):
            tracker.drain_live("s-1")
    finally:
        dispose()


def test_tracker_disposer_stops_routing(tmp_path: Path) -> None:
    """After disposal, later events are ignored and no timer fires."""
    _persistence, _handle, tracker, path = _tracker_setup(tmp_path, delay=0.01)
    ctx = Context()
    dispose = tracker.install(ctx)
    ctx.emit(
        "session/event",
        _fake_session("s-1"),
        _session_event("s-1", "turn/start", 0, {"turn": 1}),
    )
    dispose()
    dispose()
    ctx.emit(
        "session/event",
        _fake_session("s-1"),
        _session_event("s-1", "turn/start", 1, {"turn": 1}),
    )
    tracker.drain_live("s-1")
    assert _stored_seqs(path) == [0]
    time.sleep(0.1)
    assert _stored_seqs(path) == [0]


def test_tracker_double_install_raises_and_reinstall_works(tmp_path: Path) -> None:
    """A second live install raises; a stale disposer cannot kill a new one."""
    _persistence, _handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    ctx = Context()
    first_dispose = tracker.install(ctx)
    with pytest.raises(ValueError, match="already installed"):
        tracker.install(ctx)
    first_dispose()
    second_dispose = tracker.install(ctx)
    try:
        ctx.emit(
            "session/event",
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 0, {"turn": 1}),
        )
        first_dispose()
        ctx.emit(
            "session/event",
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 1, {"turn": 1}),
        )
        tracker.drain_live("s-1")
        assert _stored_seqs(path) == [0, 1]
    finally:
        second_dispose()


class _StubContext:
    """A context whose `on` returns no unregister call."""

    def __init__(self) -> None:
        self.listeners: dict[str, list[Callable[..., None]]] = {}

    def on(self, event: str, callback: Callable[..., None]) -> None:
        """Record the listener; nothing is returned to unregister with."""
        self.listeners.setdefault(event, []).append(callback)

    def emit(self, event: str, *args: Any) -> None:
        """Call every listener for the event, in registration order."""
        for callback in list(self.listeners.get(event, [])):
            callback(*args)


def test_tracker_disposer_works_without_unregister_api(tmp_path: Path) -> None:
    """Where the context offers no unregister call, disposal stops routing."""
    _persistence, _handle, tracker, path = _tracker_setup(tmp_path, delay=10.0)
    ctx = _StubContext()
    dispose = tracker.install(ctx)
    try:
        ctx.emit(
            "session/event",
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 0, {"turn": 1}),
        )
        dispose()
        ctx.emit("session/flush", _fake_session("s-1"))
        ctx.emit("session/disposed", _fake_session("s-1"))
        ctx.emit(
            "session/event",
            _fake_session("s-1"),
            _session_event("s-1", "turn/start", 1, {"turn": 1}),
        )
        tracker.drain_live("s-1")
        assert _stored_seqs(path) == [0]
    finally:
        # Already disposed; the second call proves idempotency here too.
        dispose()
