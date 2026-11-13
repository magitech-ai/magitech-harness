"""Two-process ownership proof for the JSONL backend.

A child process holds a write handle while the parent proves a second
write open is refused and a read open still works; the child then exits
without closing, and the parent proves a write open succeeds again —
the lease is released by the kernel on process death, not by an
in-process claim. Spawn context throughout, so the proof runs on
Windows too. Hermetic: temporary directories only, no network.
"""

import multiprocessing
import sys
from pathlib import Path
from typing import Any

import pytest

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.persistence_jsonl import (
    JsonlSessionPersistence,
    SessionAlreadyOwnedError,
)


def _child_hold_write(root: str, session_id: str, ready: Any, release: Any) -> None:
    """Open a write handle, hold it, then die without closing it.

    The handle is deliberately never closed: ``os._exit`` skips all
    cleanup, so only the kernel releasing the lease on process death
    can let the parent open the session for writing afterwards.
    """
    import os

    persistence = JsonlSessionPersistence(Path(root))
    persistence.open(session_id, "write")
    ready.set()
    release.wait(timeout=30)
    os._exit(0)


def test_write_ownership_released_on_process_death(tmp_path: Path) -> None:
    """A dead writer's lease frees the session; a live one blocks writers."""
    root = tmp_path / "root"
    persistence = JsonlSessionPersistence(root)
    handle = persistence.create(
        {
            "version": 4,
            "id": "s-1",
            "createdAt": 1000,
            "isSeeded": False,
            "delegationDepth": 0,
        }
    )
    handle.append(
        [
            SessionEvent(
                id="s-1:0",
                session_id="s-1",
                type="turn/start",
                timestamp=1.0,
                payload={"turn": 1},
                seq=0,
            )
        ]
    )
    handle.flush()
    handle.close()

    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    # Spawn inherits sys.path; the repo root must be on it so the child
    # can import this module by name to resolve the child entry point.
    repo_root = str(Path(__file__).resolve().parents[2])
    added_to_path = repo_root not in sys.path
    if added_to_path:
        sys.path.insert(0, repo_root)
    child = ctx.Process(
        target=_child_hold_write, args=(str(root), "s-1", ready, release)
    )
    child.start()
    try:
        assert ready.wait(timeout=30), "child never claimed the write handle"
        with pytest.raises(SessionAlreadyOwnedError):
            persistence.open("s-1", "write")
        reader = persistence.open("s-1", "read")
        _, events = reader.read()
        assert [event.seq for event in events] == [0]
        reader.close()
    finally:
        release.set()
        child.join(timeout=30)
        if child.is_alive():
            child.terminate()
            child.join(timeout=10)
        if added_to_path:
            sys.path.remove(repo_root)
    assert child.exitcode == 0
    writer = persistence.open("s-1", "write")
    writer.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
