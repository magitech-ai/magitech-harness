"""Swapped-row injection probes for create_harness(profile=...)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from animas import Context, Service

from magitech_harness import create_harness
from magitech_harness.fs.local import LocalFileSystem
from magitech_harness.profiles.merge import merge_bundle_rows

_STUB_MODULE = "tests.unit.magitech_harness.test_create_harness_profile"


class StubFileSystem(LocalFileSystem):
    """Marker subclass proving a swapped row is what gets injected."""


class ExtraService(Service):
    """Minimal appendable component for the new-row probe."""

    def __init__(self, ctx: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(ctx, "test-extra")
        self.config = config


def _swap_fs_profile() -> list[dict[str, Any]]:
    return [
        {
            "id": "fs",
            "name": f"{_STUB_MODULE}:StubFileSystem",
            "config": {},
        }
    ]


async def test_profile_swaps_fs_row() -> None:
    """A profile row with a known base id replaces the base row in place."""
    ctx = create_harness(profile=_swap_fs_profile())
    await ctx.flush()
    fs = ctx.get("fs", None, strict=False)
    assert isinstance(fs, StubFileSystem)
    assert type(fs) is StubFileSystem


async def test_profile_config_override_takes_effect() -> None:
    """Overriding an existing row's config applies the new config."""
    tmp = Path("/tmp/opencode/p5-cwd")
    ctx = create_harness(
        profile=[
            {
                "id": "fs",
                "name": "magitech_harness.fs.local:LocalFileSystem",
                "config": {"cwd": str(tmp)},
            }
        ]
    )
    await ctx.flush()
    fs = ctx.get("fs", None, strict=False)
    assert isinstance(fs, LocalFileSystem)
    # resolve() uses the configured cwd when no per-call cwd is given.
    target = await fs.resolve("file.txt")
    assert str(tmp) in target["targetKey"]


async def test_profile_appends_new_row() -> None:
    """A profile row with a new id mounts in addition to base rows."""
    ctx = create_harness(
        profile=[
            {
                "id": "test-extra",
                "name": f"{_STUB_MODULE}:ExtraService",
                "config": {},
            }
        ]
    )
    await ctx.flush()
    extra = ctx.get("test-extra", None, strict=False)
    assert isinstance(extra, ExtraService)


async def test_profile_duplicate_id_in_profile_raises() -> None:
    """Duplicate ids within one profile fail loud."""
    row = {"id": "fs", "name": "magitech_harness.fs.local:LocalFileSystem"}
    with pytest.raises(ValueError, match="duplicate"):
        create_harness(profile=[row, dict(row)])


async def test_profile_row_without_id_appends_when_named() -> None:
    """An id-less named row appends; a non-string id fails loud."""
    ctx = create_harness(
        profile=[{"name": "magitech_harness.fs.local:LocalFileSystem"}]
    )
    await ctx.flush()
    assert isinstance(ctx.get("fs", None, strict=False), LocalFileSystem)
    with pytest.raises(ValueError, match="'id'"):
        create_harness(
            profile=[{"id": 42, "name": "magitech_harness.fs.local:LocalFileSystem"}]
        )


async def test_profile_unknown_row_missing_name_raises() -> None:
    """An unknown profile row without a name fails loud."""
    with pytest.raises(ValueError, match="name"):
        create_harness(profile=[{"id": "mystery-row"}])


async def test_profile_disable_base_row() -> None:
    """A disable-only profile row disables the base row without removing name."""
    ctx = create_harness(profile=[{"id": "compaction", "disabled": True}])
    await ctx.flush()
    assert ctx.get("compaction", None, strict=False) is None
    # The base row kept its service: disabling is not whole-row replacement.
    ctx2 = create_harness()
    await ctx2.flush()
    assert ctx2.get("compaction", None, strict=False) is not None


def test_merge_bundle_rows_preserves_base_expression_nodes() -> None:
    """Merging must not stringify base conditional-disable expressions."""
    expr = {"__jsExpr": "process.platform === 'win32'"}
    base_rows: list[dict[str, Any]] = [
        {"id": "pwsh-local", "name": "mod:Pwsh", "disabled": expr, "config": {}},
        {"id": "fs", "name": "mod:Fs", "config": {}},
    ]
    merged = merge_bundle_rows(base_rows, [{"id": "fs", "config": {"cwd": "/nowhere"}}])
    assert merged[0]["disabled"] == expr
    assert merged[0]["name"] == "mod:Pwsh"
    assert merged[1]["name"] == "mod:Fs"
    assert merged[1]["config"] == {"cwd": "/nowhere"}


def test_merge_bundle_rows_appends_unknown_ids_in_profile_order() -> None:
    """Unknown ids append after base rows in profile order."""
    merged = merge_bundle_rows(
        [{"id": "a", "name": "mod:A", "config": {}}],
        [
            {"id": "c", "name": "mod:C", "config": {}},
            {"id": "b", "name": "mod:B", "config": {}},
        ],
    )
    assert [row["id"] for row in merged] == ["a", "c", "b"]


def test_merge_bundle_rows_rejects_profile_duplicates_and_bad_rows() -> None:
    """The merge helper fails loud on duplicate ids and missing fields."""
    base = [{"id": "a", "name": "mod:A", "config": {}}]
    with pytest.raises(ValueError, match="duplicate"):
        merge_bundle_rows(
            base, [{"id": "x", "name": "m:X"}, {"id": "x", "name": "m:X2"}]
        )
    with pytest.raises(ValueError, match="'id'"):
        merge_bundle_rows(base, [{"id": 42, "name": "m:X"}])
    with pytest.raises(ValueError, match="declare a 'name'"):
        merge_bundle_rows(base, [{"id": "x"}])
    with pytest.raises(ValueError, match="declare a 'name'"):
        merge_bundle_rows(base, [{}])


def test_merge_bundle_rows_rejects_non_dict_profile_row() -> None:
    """A non-mapping profile row fails loud."""
    with pytest.raises(TypeError, match="mappings"):
        merge_bundle_rows([], [42])


def test_merge_bundle_rows_rejects_known_id_cleared_name() -> None:
    """Overriding a known id cannot clear its service name."""
    with pytest.raises(ValueError, match="resolvable service name"):
        merge_bundle_rows(
            [{"id": "fs", "name": "mod:Fs", "config": {}}], [{"id": "fs", "name": ""}]
        )


def test_merge_bundle_rows_tolerates_base_rows_without_ids() -> None:
    """Base rows without string ids pass through untouched."""
    merged = merge_bundle_rows(
        ["bare-string-row", {"id": "fs", "name": "mod:Fs", "config": {}}],
        [{"id": "fs", "config": {"cwd": "/x"}}],
    )
    assert merged[0] == "bare-string-row"
    assert merged[1]["config"] == {"cwd": "/x"}


def test_profile_from_yaml_path(tmp_path: Path) -> None:
    """A str/Path profile source is read from disk."""
    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text("- id: compaction\n  disabled: true\n", encoding="utf-8")
    ctx = create_harness(profile=str(profile_file))
    assert ctx.get("compaction", None, strict=False) is None


def test_profile_components_document_and_single_row_dict() -> None:
    """A dict profile is either a components document or a single row."""
    ctx = create_harness(
        profile={
            "components": [
                {
                    "id": "fs",
                    "name": "magitech_harness.fs.local:LocalFileSystem",
                    "config": {},
                }
            ]
        }
    )
    assert isinstance(ctx.get("fs", None, strict=False), LocalFileSystem)

    single = {"id": "compaction", "disabled": True}
    ctx2 = create_harness(profile=dict(single))
    assert ctx2.get("compaction", None, strict=False) is None


def test_profile_invalid_type_raises() -> None:
    """A non-path/list/dict profile fails loud."""
    with pytest.raises(TypeError, match="profile must be"):
        create_harness(profile=42)


def test_profile_path_document_must_be_list(tmp_path: Path) -> None:
    """A profile file whose document is not a row list fails loud."""
    profile_file = tmp_path / "bad.yaml"
    profile_file.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a list"):
        create_harness(profile=str(profile_file))


def test_profile_components_key_must_be_list() -> None:
    """A components-document profile with a non-list components key fails loud."""
    with pytest.raises(ValueError, match="'components' must be a list"):
        create_harness(profile={"components": "nope"})


async def test_patch_overlays_apply_after_profile_in_order() -> None:
    """Extra patch overlays resolve after the profile, in argv order."""
    # Profile sets cwd to A; first patch overrides to B; second patch to C.
    profile = [
        {
            "id": "fs",
            "name": "magitech_harness.fs.local:LocalFileSystem",
            "config": {"cwd": "/layer-a"},
        }
    ]
    patch_b = [
        {
            "id": "fs",
            "name": "magitech_harness.fs.local:LocalFileSystem",
            "config": {"cwd": "/layer-b"},
        }
    ]
    patch_c = [
        {
            "id": "fs",
            "name": "magitech_harness.fs.local:LocalFileSystem",
            "config": {"cwd": "/layer-c"},
        }
    ]
    ctx = create_harness(profile=profile, patches=[patch_b, patch_c])
    await ctx.flush()
    fs = ctx.get("fs", None, strict=False)
    assert isinstance(fs, LocalFileSystem)
    target = await fs.resolve("file.txt")
    assert "layer-c" in Path(target["targetKey"]).parts


async def test_patches_empty_is_noop() -> None:
    """An empty patches list behaves as no overlays."""
    ctx = create_harness(patches=[])
    await ctx.flush()
    assert ctx.get("fs", None, strict=False) is not None


def test_patch_path_document_must_be_list(tmp_path: Path) -> None:
    """A patch file whose document is not a row list fails loud."""
    patch_file = tmp_path / "bad-patch.yaml"
    patch_file.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(TypeError, match="must be a list"):
        create_harness(patches=[str(patch_file)])
