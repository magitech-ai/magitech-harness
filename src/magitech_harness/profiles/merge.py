"""Bundle-row merge for create_harness(profile=...) swapped-row injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anima.jsexpr import load_yaml

__all__ = ["merge_bundle_rows", "profile_rows_from_source"]


def profile_rows_from_source(profile: Any) -> list[Any]:
    """Normalize a profile source (path / list / dict) into a row list.

    A string or Path is read as YAML. A dict is either a ``{"components":
    [...]}`` document or a single row. A list is the row list itself.
    """
    if isinstance(profile, (str, Path)):
        with open(Path(profile), encoding="utf-8") as stream:
            loaded = load_yaml(stream)
        if not isinstance(loaded, list):
            raise TypeError("profile document must be a list of component rows")
        return list(loaded)
    if isinstance(profile, dict):
        components = profile.get("components")
        if components is not None:
            if not isinstance(components, list):
                raise ValueError("profile 'components' must be a list")
            return list(components)
        return [profile]
    if isinstance(profile, list):
        return list(profile)
    raise TypeError("profile must be a path, list, or dict")


def merge_bundle_rows(base_rows: list[Any], profile_rows: list[Any]) -> list[Any]:
    """Merge profile rows over base bundle rows by id.

    A profile row with a known base id overlays its keys onto a copy of the
    base row, preserving the base slot position and any base fields the
    profile does not set (including conditional-disable expression nodes).
    Rows with unknown ids append after the base rows in profile order; an
    id-less named row (including a bare component-name string) also appends.
    Duplicate profile ids, non-string ids, and rows without a resolvable
    service name fail loud.
    """
    merged: list[Any] = [
        dict(row) if isinstance(row, dict) else row for row in base_rows
    ]
    index_by_id: dict[str, int] = {}
    for position, row in enumerate(merged):
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            index_by_id[row["id"]] = position

    seen: set[str] = set()
    appended: list[Any] = []
    for row in profile_rows:
        if isinstance(row, str):
            # Loader semantics: a bare string row names a component to mount.
            row = {"name": row}
        if not isinstance(row, dict):
            raise TypeError("profile rows must be mappings with an 'id'")
        row_id = row.get("id")
        if row_id is None:
            # An id-less row (e.g. a bare string name) always appends.
            if not row.get("name"):
                raise ValueError("profile row without an 'id' must declare a 'name'")
            appended.append(dict(row))
            continue
        if not isinstance(row_id, str) or not row_id:
            raise ValueError("profile rows must carry a non-empty string 'id'")
        if row_id in seen:
            raise ValueError(f"duplicate profile row id '{row_id}'")
        seen.add(row_id)

        if row_id in index_by_id:
            position = index_by_id[row_id]
            combined = dict(merged[position])
            for key, value in row.items():
                if key == "id":
                    continue
                combined[key] = value
            if not combined.get("name"):
                raise ValueError(
                    f"profile row '{row_id}' must keep a resolvable service name"
                )
            merged[position] = combined
        else:
            if not row.get("name"):
                raise ValueError(f"profile row '{row_id}' must declare a 'name'")
            appended.append(dict(row))

    return merged + appended
