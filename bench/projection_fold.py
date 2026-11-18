"""Workload: repeated surface fold over a fixed event list (pure, no I/O)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from bench._synthetic import write_session
from bench._timing import Timing, measure
from magitech_harness.session.store import Session
from magitech_harness.session.surface import fold_surface

_TURNS = 120  # ~245 events folded per iteration


def projection_fold(*, warmup: int, repeat: int) -> Timing:
    with tempfile.TemporaryDirectory(prefix="bench-fold-") as tmp:
        path = write_session(Path(tmp), _TURNS)
        events = list(Session.load(path).events)

        def run() -> None:
            fold_surface(events)

        return measure(
            "projection-fold/surface",
            run,
            iterations=1,
            unit="folds",
            warmup=warmup,
            repeat=repeat,
        )
