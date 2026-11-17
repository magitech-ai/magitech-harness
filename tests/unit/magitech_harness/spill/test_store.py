"""Unit tests for the spill storage mechanics (saveTextFile, encodeSegment, sessionDir, privateRoot)."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

from magitech_harness.spill.store import (
    encodeSegment,
    isErrno,
    privateRoot,
    saveTextFile,
    sessionDir,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("web_fetch.txt", "web_fetch.txt"),
        ("a-B_9.z", "a-B_9.z"),
        ("../etc/passwd", "..~002Fetc~002Fpasswd"),
        ("a/b", "a~002Fb"),
        ("~", "~007E"),
        (".", "~002E"),
        ("..", "~002E~002E"),
        ("", "~"),
    ],
)
def test_encode_segment(raw: str, expected: str) -> None:
    """encodeSegment produces one safe path segment from untrusted input."""
    assert encodeSegment(raw) == expected


def test_session_dir_is_stable_per_session() -> None:
    """sessionDir is deterministic and session-scoped under root."""
    dir1 = sessionDir("/spill", "sess-1")
    assert dir1 == sessionDir("/spill", "sess-1")
    assert Path(dir1).parent == Path("/spill")
    assert Path(dir1).name.startswith("session-")
    assert len(Path(dir1).name.split("-")[1]) == 12
    assert sessionDir("/spill", "sess-2") != dir1


def test_save_text_file_writes_under_session_dir(tmp_path: Path) -> None:
    """saveTextFile writes content under the session dir and reports bytes."""
    saved = saveTextFile(
        {
            "root": str(tmp_path),
            "sessionId": "sess-1",
            "suggestedName": "r.txt",
            "content": "héllo",
        }
    )
    assert Path(saved["path"]).read_text(encoding="utf-8") == "héllo"
    assert saved["bytes"] == len("héllo".encode())
    assert str(Path(saved["path"]).parent) == sessionDir(str(tmp_path), "sess-1")
    assert Path(saved["path"]).name.endswith("-r.txt")


def test_save_text_file_sanitizes_traversal_name(tmp_path: Path) -> None:
    """A traversal-shaped suggested name becomes one safe leaf segment."""
    saved = saveTextFile(
        {
            "root": str(tmp_path),
            "sessionId": "sess-1",
            "suggestedName": "../../evil",
            "content": "x",
        }
    )
    assert str(Path(saved["path"]).parent) == sessionDir(str(tmp_path), "sess-1")
    assert "/.." not in saved["path"]


def test_save_text_file_creates_owner_only_permissions(tmp_path: Path) -> None:
    """The session dir and file are created owner-only (0700 / 0600)."""
    saved = saveTextFile(
        {
            "root": str(tmp_path),
            "sessionId": "sess-1",
            "suggestedName": "r.txt",
            "content": "x",
        }
    )
    dir_stat = os.stat(Path(saved["path"]).parent)
    file_stat = os.stat(saved["path"])
    if os.name != "nt":  # Windows does not support Unix file modes
        assert (dir_stat.st_mode & 0o777) == 0o700
        assert (file_stat.st_mode & 0o777) == 0o600


def test_save_text_file_distinct_paths_for_same_name(tmp_path: Path) -> None:
    """Two saves of the same name produce distinct paths."""
    a = saveTextFile(
        {
            "root": str(tmp_path),
            "sessionId": "sess-1",
            "suggestedName": "r.txt",
            "content": "a",
        }
    )
    b = saveTextFile(
        {
            "root": str(tmp_path),
            "sessionId": "sess-1",
            "suggestedName": "r.txt",
            "content": "b",
        }
    )
    assert a["path"] != b["path"]


def test_private_root_is_stable_and_absolute() -> None:
    """privateRoot returns a stable absolute directory under the temp dir."""
    first = privateRoot()
    assert os.path.isabs(first)
    assert privateRoot() == first


def test_isErrno_matches_system_error_code() -> None:
    """isErrno matches errno codes and rejects non-matches."""
    err = OSError(errno.ENOENT, "boom")
    assert isErrno(err, "ENOENT") is True
    assert isErrno(err, "EPERM") is False
    assert isErrno("not an error", "ENOENT") is False
    assert isErrno(OSError("no code"), "ENOENT") is False


@pytest.mark.asyncio
async def test_save_text_file_name_collision_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """saveTextFile retries on FileExistsError (name collision)."""
    real_open = os.open
    call_count = [0]

    def fake_open(path, flags, mode=0o600, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise FileExistsError(17, "File exists", path)
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", fake_open)

    saved = saveTextFile(
        {
            "root": str(tmp_path),
            "sessionId": "sess-1",
            "suggestedName": "r.txt",
            "content": "hello",
        }
    )
    assert Path(saved["path"]).read_text(encoding="utf-8") == "hello"
    assert call_count[0] == 2  # First call failed, second succeeded


@pytest.mark.asyncio
async def test_save_text_file_write_failure_propagates(tmp_path: Path) -> None:
    """saveTextFile propagates write errors."""
    from unittest.mock import patch

    def fake_write(fd, data):
        raise OSError("write failed")

    with (
        patch("magitech_harness.spill.store._write_to_fd", fake_write),
        pytest.raises(OSError, match="write failed"),
    ):
        saveTextFile(
            {
                "root": str(tmp_path),
                "sessionId": "sess-1",
                "suggestedName": "r.txt",
                "content": "hello",
            }
        )
