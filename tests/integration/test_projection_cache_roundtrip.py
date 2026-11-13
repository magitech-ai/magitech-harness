"""Two-process roundtrip and crash tests for the persisted projection cache.

The child driver (``_projcache_crash_child.py``) writes Scenario A through a
real service in its own process; the parent then cold-reads the same root.
The kill tests prove the cache never claims a seq beyond the durable log,
even when the process dies mid-write.
"""

import json
import signal
import subprocess
import sys
from pathlib import Path

from anima import Context

from magitech_harness.session.service import SessionService

_CHILD = Path(__file__).with_name("_projcache_crash_child.py")


def _run_child(root: Path, mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_CHILD), str(root), "s1", mode],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _record_rows(root: Path) -> dict[str, int]:
    doc = json.loads(
        (root / "session_projcache" / "sessions" / "s1.json").read_text(
            encoding="utf-8"
        )
    )
    return {key: row["seq"] for key, row in doc["record"]["rows"].items()}


def _log_events(root: Path) -> list[dict]:
    service = SessionService(Context(), {"base_storage_dir": root})
    path = service.location("s1")
    assert path is not None
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["type"] == "session"
    return [json.loads(line) for line in lines[1:]]


def _assert_partial_golden(snapshot: dict) -> None:
    """The log through turn/end folds to Scenario A minus the title/cwd tail."""
    assert snapshot["as_of_seq"] == 7
    values = snapshot["values"]
    assert values["title"] is None
    assert values["turnOutline"] == [
        {"turn": 1, "seq": 0, "prompt": "Hello world", "response": "Hi there"}
    ]
    assert values["workingDirectory"] is None
    stats = values["sessionStats"]
    assert stats["turns"] == 1
    assert stats["steps"] == 1
    assert stats["ttft_steps"] == 1
    assert stats["decode_tokens"] == 5


def _assert_full_golden(snapshot: dict) -> None:
    """The complete Scenario A log folds to the golden values."""
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


def test_child_roundtrip_cold_read(tmp_path: Path) -> None:
    """A child-written session cold-reads correctly in a fresh process."""
    result = _run_child(tmp_path, "roundtrip")
    assert result.returncode == 0, result.stderr
    assert _record_rows(tmp_path)["title"] == 9
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    snapshot = service.read_projections("s1")
    assert snapshot is not None
    _assert_full_golden(snapshot)
    loaded = service.load("s1")
    assert loaded is not None
    _assert_full_golden(service.projections.snapshot(loaded))
    assert service.projection_cache is not None
    service.projection_cache.dispose()


def test_kill_at_put_cache_never_ahead_of_log(tmp_path: Path) -> None:
    """A SIGKILL at the cache write leaves the cache behind the log."""
    result = _run_child(tmp_path, "kill-at-put")
    assert result.returncode == -signal.SIGKILL
    events = _log_events(tmp_path)
    assert len(events) == 8
    assert events[-1]["seq"] == 7
    assert events[-1]["type"] == "turn/end"
    # The create-point record is the only durable cache row.
    assert _record_rows(tmp_path)["title"] == -1
    assert _record_rows(tmp_path)["title"] <= events[-1]["seq"]
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    loaded = service.load("s1")
    assert loaded is not None
    _assert_partial_golden(service.projections.snapshot(loaded))
    # The load's create-point write folds the durable log into the cache.
    assert _record_rows(tmp_path)["title"] == 7
    assert service.projection_cache is not None
    service.projection_cache.dispose()


def test_kill_mid_put_record_intact_and_tmp_ignored(tmp_path: Path) -> None:
    """A SIGKILL mid-rename leaves the old record intact and a stray temp."""
    result = _run_child(tmp_path, "kill-mid-put")
    assert result.returncode == -signal.SIGKILL
    sessions_dir = tmp_path / "session_projcache" / "sessions"
    leftovers = list(sessions_dir.glob("*.tmp"))
    assert len(leftovers) == 1
    # The old record still parses: no torn document.
    assert _record_rows(tmp_path)["title"] == -1
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    loaded = service.load("s1")
    assert loaded is not None
    _assert_partial_golden(service.projections.snapshot(loaded))
    # The next open ignores the leftover temp instead of choking on it.
    assert list(sessions_dir.glob("*.tmp")) == leftovers
    assert _record_rows(tmp_path)["title"] == 7
    assert service.projection_cache is not None
    service.projection_cache.dispose()
