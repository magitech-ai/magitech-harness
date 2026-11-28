"""Named profiles, boot-free config dumps, and profile component lists.

This is the Python counterpart of the dsh launcher's profile plumbing: a named
profile is a directory under ``$MTH_HOME/profiles/<name>`` holding a YAML row
list and a marketplace component list. Boot still goes through ``create_harness``;
these helpers only resolve inputs, compose rows for inspection, and edit the
component list. No `!!js` evaluator is introduced (ADR 0022).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from animas.profile_yaml import load_yaml

from magitech_harness.profiles.merge import merge_bundle_rows, profile_rows_from_source

HOME_ENV = "MTH_HOME"
PROFILE_DIR = "profiles"
PROFILE_FILE = "profile.yaml"
COMPONENTS_FILE = "components.yaml"
TEMPLATES = Path(__file__).parent / "templates"
SHIPPED_TEMPLATES = frozenset({"base", "headless"})
RESERVED_NAMES = frozenset({"desktop"})


def profiles_root() -> Path:
    """Return the profile root, defaulting to ``~/.mth/profiles``."""
    configured = os.environ.get(HOME_ENV)
    base = Path(configured) if configured else Path.home() / ".mth"
    return base / PROFILE_DIR


def profile_dir(name: str) -> Path:
    """Resolve a profile name to its directory, rejecting reserved names."""
    if not name or not name.strip():
        raise ValueError("profile name must not be blank")
    if name.lower() in RESERVED_NAMES:
        raise ValueError(f'profile "{name}" is reserved')
    return profiles_root() / name


def _copy_template(name: str, destination: Path, *, overwrite: bool) -> None:
    source = TEMPLATES / name
    if not source.is_dir():
        raise ValueError(f"unknown default profile template: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    if overwrite:
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return
    for item in source.iterdir():
        if not item.is_file() or (destination / item.name).exists():
            continue
        shutil.copy2(item, destination / item.name)


def prepare_profile(name: str, from_default: str | None = None) -> Path:
    """Create or reset a named profile and return its directory."""
    destination = profile_dir(name)
    if from_default is not None:
        if not from_default.strip():
            raise ValueError("--from-default-profile needs a name")
        _copy_template(from_default, destination, overwrite=True)
    elif not (destination / PROFILE_FILE).exists():
        if destination.exists():
            # A profile directory created by `mth plugin` already exists; give
            # it the default row list without discarding its component list.
            _copy_template("base", destination, overwrite=False)
        elif name in SHIPPED_TEMPLATES:
            _copy_template(name, destination, overwrite=False)
        else:
            raise ValueError(f"profile not found: {name}")
    return destination


def resolve_profile_source(value: str | None, from_default: str | None = None) -> Any:
    """Resolve a CLI ``--profile`` value to a path or raw YAML row source.

    A file path or YAML document keeps the existing behavior; a bare name
    resolves through the profile directory, initializing shipped templates on
    first use.
    """
    if value is None:
        return None
    candidate = Path(value)
    if candidate.exists() or "/" in value or value.endswith((".yaml", ".yml")):
        return value
    return prepare_profile(value, from_default) / PROFILE_FILE


def _load_rows(source: Any) -> list[Any]:
    if source is None:
        return []
    return profile_rows_from_source(source)


def _base_rows() -> list[Any]:
    with (Path(__file__).parent / "base.yaml").open(encoding="utf-8") as stream:
        loaded = load_yaml(stream)
    return list(loaded) if isinstance(loaded, list) else []


def compose_rows(profile: Any, patches: list[Any] | None = None) -> list[Any]:
    """Compose base, profile, and patch rows with the boot-time merge rules."""
    rows = _base_rows()
    rows = merge_bundle_rows(rows, _load_rows(profile))
    for patch in patches or []:
        rows = merge_bundle_rows(rows, _load_rows(patch))
    return rows


def dump_config(
    profile: Any = None,
    patches: list[Any] | None = None,
    *,
    default_only: bool = False,
) -> dict[str, Any]:
    """Return a boot-free composition report with one layer per source."""
    layers: list[dict[str, Any]] = [{"label": "base", "rows": _base_rows()}]
    if not default_only:
        if profile is not None:
            layers.append({"label": str(profile), "rows": _load_rows(profile)})
        for patch in patches or []:
            layers.append({"label": str(patch), "rows": _load_rows(patch)})
    rows = _base_rows() if default_only else compose_rows(profile, patches)
    return {"schema": "mth-config-dump/v1", "layers": layers, "composed": rows}


def dump_config_schema() -> dict[str, Any]:
    """Return JSON Schema for one profile row and for a whole patch list."""
    row = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "id": {"type": "string", "description": "Stable row identity for overlays"},
            "name": {"type": "string", "description": "Component mount target"},
            "config": {"type": "object"},
            "enabled_on": {"type": "array", "items": {"type": "string"}},
            "disabled_on": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "mth profile rows",
        "row": row,
        "patchList": {"type": "array", "items": row},
        "x-mth": {"complete": True, "diagnostics": []},
    }


def component_path(name: str) -> Path:
    """Return the marketplace component list for a named profile."""
    return profile_dir(name) / COMPONENTS_FILE


def read_components(name: str) -> list[str]:
    """Read a profile's component list; an absent file means no components."""
    path = component_path(name)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        loaded = load_yaml(stream)
    if loaded is None:
        return []
    if not isinstance(loaded, list) or not all(
        isinstance(item, str) for item in loaded
    ):
        raise ValueError(f"{path} must be a list of component names")
    return list(loaded)


def write_components(name: str, components: list[str]) -> Path:
    """Write a profile's component list atomically with an owner-only file."""
    path = component_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(json.dumps(components, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def edit_components(name: str, action: str, component: str | None = None) -> list[str]:
    """Apply a list/add/remove/show action to a profile's component list."""
    components = read_components(name)
    if action not in ("list", "add", "remove", "show"):
        raise ValueError(f"unknown component action: {action}")
    if action == "list":
        return components
    if component is None or not component.strip():
        raise ValueError(f"component {action} needs a component name")
    if action == "add":
        if component not in components:
            components.append(component)
    elif action == "remove":
        components = [item for item in components if item != component]
    else:
        if component not in components:
            raise ValueError(f"component not installed in profile: {component}")
        return [component]
    write_components(name, components)
    return components
