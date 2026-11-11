"""Unit tests for ComponentManifest schema and validation."""

import pytest
from pydantic import ValidationError

from magitech_harness.components.manifest import ComponentManifest


def test_component_manifest_valid() -> None:
    """Verifies parsing a valid component manifest."""
    data = {
        "name": "fs-tools",
        "version": "0.1.0",
        "description": "Filesystem tools component",
        "entry_point": "component:apply",
        "types": ["Tool"],
        "python_deps": [],
        "enabled": True,
    }
    manifest = ComponentManifest.model_validate(data)
    assert manifest.name == "fs-tools"
    assert manifest.version == "0.1.0"
    assert manifest.entry_point == "component:apply"
    assert manifest.types == ["Tool"]
    assert manifest.enabled is True


def test_component_manifest_defaults() -> None:
    """Verifies default values for optional manifest fields."""
    manifest = ComponentManifest(
        name="minimal-component",
        version="1.0.0",
        entry_point="main:component",
    )
    assert manifest.description == ""
    assert manifest.types == []
    assert manifest.python_deps == []
    assert manifest.hooks == []
    assert manifest.requires_services == []
    assert manifest.recommended_components == []
    assert manifest.enabled is True


def test_component_manifest_missing_required() -> None:
    """Verifies validation error when required fields are missing."""
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate({"name": "incomplete"})
