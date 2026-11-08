"""Unit tests for PluginManifest schema and validation."""

import pytest
from pydantic import ValidationError

from magitech_harness.plugins.manifest import PluginManifest


def test_plugin_manifest_valid() -> None:
    """Verifies parsing a valid plugin manifest."""
    data = {
        "name": "fs-tools",
        "version": "0.1.0",
        "description": "Filesystem tools plugin",
        "entry_point": "plugin:apply",
        "types": ["Tool"],
        "python_deps": [],
        "enabled": True,
    }
    manifest = PluginManifest.model_validate(data)
    assert manifest.name == "fs-tools"
    assert manifest.version == "0.1.0"
    assert manifest.entry_point == "plugin:apply"
    assert manifest.types == ["Tool"]
    assert manifest.enabled is True


def test_plugin_manifest_defaults() -> None:
    """Verifies default values for optional manifest fields."""
    manifest = PluginManifest(
        name="minimal-plugin",
        version="1.0.0",
        entry_point="main:plugin",
    )
    assert manifest.description == ""
    assert manifest.types == []
    assert manifest.python_deps == []
    assert manifest.hooks == []
    assert manifest.enabled is True


def test_plugin_manifest_missing_required() -> None:
    """Verifies validation error when required fields are missing."""
    with pytest.raises(ValidationError):
        PluginManifest.model_validate({"name": "incomplete"})
