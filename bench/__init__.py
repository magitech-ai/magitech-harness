"""Python-native benchmark suite for the harness.

Run all workloads with ``uv run python -m bench``. This package lives outside
``src/`` so it is not part of the shipped wheel and is exempt from the
``magitech_harness`` coverage gate. It uses only the standard library and the
installed harness — see ``bench/README.md`` for methodology and the recorded
baseline.
"""

from __future__ import annotations

__all__: list[str] = []
