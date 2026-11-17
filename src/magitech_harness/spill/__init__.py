"""Spill storage subsystem.

Text spill storage service (`ctx.spillStore`) with a local filesystem
backend. Stores oversized text outside the model context and returns an
opaque locator with retrieval guidance.

Reference: ``packages/spill/`` in the parity monorepo.
"""

from magitech_harness.spill.local import LocalSpillStore
from magitech_harness.spill.service import (
    SaveTextSpill,
    SpillLocator,
    SpillOwner,
    SpillRef,
    SpillSource,
    SpillStore,
)

__all__ = [
    "Config",
    "LocalSpillStore",
    "SaveTextSpill",
    "SpillLocator",
    "SpillOwner",
    "SpillRef",
    "SpillSource",
    "SpillStore",
]
