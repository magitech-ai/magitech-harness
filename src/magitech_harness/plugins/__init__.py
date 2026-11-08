"""Portable plugin system for Magitech Harness."""

from magitech_harness.plugins.layers import ConfigLayer, LayerResolutionService
from magitech_harness.plugins.manifest import PluginManifest
from magitech_harness.plugins.service import PluginService

__all__ = [
    "ConfigLayer",
    "LayerResolutionService",
    "PluginManifest",
    "PluginService",
]
