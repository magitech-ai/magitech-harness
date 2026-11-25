"""Conformance: filesystem tools confine paths to the workspace root (#20)."""

from __future__ import annotations

from pathlib import Path

import pytest

from magitech_harness.tools.fs import handle_read_file, handle_write_file
from magitech_harness.tools.path_guard import PathEscapeError, confined_path


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "sub").mkdir()
    (workspace / "in.txt").write_text("inside", encoding="utf-8")
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")
    return workspace, secret


def test_confined_path_rejects_dotdot_and_symlink(layout: tuple[Path, Path]) -> None:
    workspace, secret = layout
    assert confined_path(workspace, "sub/../in.txt") == workspace / "in.txt"
    with pytest.raises(PathEscapeError):
        confined_path(workspace, str(secret))
    with pytest.raises(PathEscapeError):
        confined_path(workspace, "sub/../../secret.txt")
    link = workspace / "link"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):  # pragma: no cover - Windows ACL
        pytest.skip("symlinks unavailable")
    with pytest.raises(PathEscapeError):
        confined_path(workspace, str(link))


@pytest.mark.asyncio
async def test_read_file_denies_out_of_workspace(layout: tuple[Path, Path]) -> None:
    workspace, secret = layout
    result = await handle_read_file({"path": str(secret)}, workspace_root=workspace)
    assert "escapes workspace" in result
    assert "TOP-SECRET" not in result
    assert (
        await handle_read_file({"path": "in.txt"}, workspace_root=workspace) == "inside"
    )


@pytest.mark.asyncio
async def test_write_file_denies_out_of_workspace(layout: tuple[Path, Path]) -> None:
    workspace, secret = layout
    result = await handle_write_file(
        {"path": str(secret), "content": "pwned"}, workspace_root=workspace
    )
    assert "escapes workspace" in result
    assert secret.read_text(encoding="utf-8") == "TOP-SECRET"
    await handle_write_file(
        {"path": "out.txt", "content": "ok"}, workspace_root=workspace
    )
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "ok"
