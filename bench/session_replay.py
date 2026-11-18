"""Workload: cold load + replay of a synthetic v4 session log.

The session is written once (outside timing); each timed sample is a fresh
``Session.load`` + ``derive_messages`` from disk, isolating the load/replay path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bench._synthetic import write_session
from bench._timing import Timing, measure
from magitech_harness.session.store import Session

_TURNS = 120  # -> ~245 events, ~243 derived messages


def session_replay(*, warmup: int, repeat: int) -> Timing:
    with tempfile.TemporaryDirectory(prefix="bench-session-") as tmp:
        path = write_session(Path(tmp), _TURNS)

        def run() -> None:
            session = Session.load(path)
            session.derive_messages()

        timing = measure(
            "session-replay/cold-load",
            run,
            iterations=1,
            unit="sessions",
            warmup=warmup,
            repeat=repeat,
        )
    return timing
