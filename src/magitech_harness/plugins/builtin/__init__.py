"""Built-in plugins for Magitech Harness."""

from magitech_harness.plugins.builtin.fs import apply as fs_tools
from magitech_harness.plugins.builtin.truncation import (
    TruncationResult,
    format_size,
    truncate_head,
    truncate_tail,
)
from magitech_harness.plugins.builtin.truncation import (
    apply as truncation_guard,
)

__all__ = [
    "TruncationResult",
    "format_size",
    "fs_tools",
    "truncate_head",
    "truncate_tail",
    "truncation_guard",
]
