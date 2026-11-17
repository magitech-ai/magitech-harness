"""Spill storage subsystem.

Text spill storage service (`ctx.spillStore`) with a local filesystem
backend. Stores oversized text outside the model context and returns an
opaque locator with retrieval guidance.

Reference: ``packages/spill/`` in the parity monorepo.
"""

from magitech_harness.spill.local import LocalSpillStore
from magitech_harness.spill.notice import (
    format_spill_notice,
    has_spill_notice,
    is_spill_notice,
)
from magitech_harness.spill.policy import Config, SpillPolicyExec, apply, inject, name
from magitech_harness.spill.retention import RetentionResult, retain_content
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
    "RetentionResult",
    "SaveTextSpill",
    "SpillLocator",
    "SpillOwner",
    "SpillPolicyExec",
    "SpillRef",
    "SpillSource",
    "SpillStore",
    "apply",
    "format_spill_notice",
    "has_spill_notice",
    "inject",
    "is_spill_notice",
    "name",
    "retain_content",
]
