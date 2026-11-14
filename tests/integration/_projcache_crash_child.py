"""Child driver for the projection-cache crash tests (not a test module).

Run as: ``python _projcache_crash_child.py <root> <session_id> <mode>``.

Modes:
- ``roundtrip``: write Scenario A through a real service and close cleanly.
- ``kill-at-put``: SIGKILL the process on the second cache write (the
  turn/end write, which runs after the log flush in ``write()``).
- ``kill-mid-put``: SIGKILL the process inside the second atomic rename
  (the turn/end write), after the temp file was written and fsynced.
"""

import os
import signal
import sys
from pathlib import Path

from anima import Context

from magitech_harness.session.service import SessionService


def _kill_self() -> None:
    """Terminate this process immediately with SIGKILL.

    The crash modes only run where SIGKILL exists; on platforms without it
    (Windows) the caller tests are skipped, so reaching this is a loud
    error rather than a silent no-op.
    """
    sigkill = getattr(signal, "SIGKILL", None)
    if sigkill is None:
        raise RuntimeError("SIGKILL is unavailable on this platform")
    os.kill(os.getpid(), sigkill)


def _append_golden_live(service: SessionService, session) -> None:
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
                "source": {"kind": "model", "provider": "p", "model": "m"},
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


def _kill_at_put(service: SessionService) -> None:
    """SIGKILL on the second cache write (the turn/end write)."""
    cache = service.projection_cache
    assert cache is not None
    real_put = cache.put
    calls = {"n": 0}

    def counting_put(session_id: str, identity, rows) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            _kill_self()
        real_put(session_id, identity, rows)

    cache.put = counting_put  # type: ignore[method-assign]


def _kill_mid_put() -> None:
    """SIGKILL inside the second cache rename, after the temp file lands."""
    real_replace = os.replace
    calls = {"n": 0}

    def counting_replace(src, dst) -> None:
        if str(src).endswith(".tmp") and "session_projcache" in str(src):
            calls["n"] += 1
            if calls["n"] == 2:
                _kill_self()
        real_replace(src, dst)

    os.replace = counting_replace  # type: ignore[method-assign]


def main(argv: list[str]) -> int:
    root, session_id, mode = argv[1], argv[2], argv[3]
    if mode == "kill-mid-put":
        _kill_mid_put()
    service = SessionService(Context(), {"base_storage_dir": Path(root)})
    if mode == "kill-at-put":
        _kill_at_put(service)
    session = service.create(session_id)
    _append_golden_live(service, session)
    service.close(session_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
