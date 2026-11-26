"""Permission presets: sandbox mode and approval policy as one user choice."""

from __future__ import annotations

from magitech_harness.permission.service import (
    AUTO_PRESET,
    AUTO_PRESET_SPEC,
    CUSTOM_PRESET,
    DEFAULT_PRESETS,
    RESERVED_PRESETS,
    PermissionPresetError,
    PermissionPresetService,
    PresetSpec,
    apply,
)

__all__ = [
    "AUTO_PRESET",
    "AUTO_PRESET_SPEC",
    "CUSTOM_PRESET",
    "DEFAULT_PRESETS",
    "RESERVED_PRESETS",
    "PermissionPresetError",
    "PermissionPresetService",
    "PresetSpec",
    "apply",
]
