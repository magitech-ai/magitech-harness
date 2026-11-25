"""Unit tests for the Tier 1 fs-search tools (glob + grep)."""

from __future__ import annotations

from pathlib import Path

import pytest
from animas import Context

from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.fs_search import _relative, handle_glob, handle_grep
from magitech_harness.tools.fs_search import apply as fs_search_apply
from magitech_harness.tools.service import ToolCallData, ToolService


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "a.py").write_text("alpha = 1\nneedle here\n", encoding="utf-8")
    (ws / "src" / "b.txt").write_text("needle in text\n", encoding="utf-8")
    (ws / ".hidden.py").write_text("hidden needle\n", encoding="utf-8")
    (ws / "readme.md").write_text("nothing here\n", encoding="utf-8")
    return ws


def test_glob_basename_pattern(workspace: Path) -> None:
    result = handle_glob({"pattern": "*.py"}, workspace_root=workspace)
    names = {Path(line).name for line in result.splitlines()}
    assert {"a.py", ".hidden.py"} <= names
    assert "b.txt" not in names


def test_glob_anchored_pattern(workspace: Path) -> None:
    result = handle_glob({"pattern": "src/*.py"}, workspace_root=workspace)
    assert "src/a.py" in result


def test_glob_empty_and_validation(workspace: Path) -> None:
    assert (
        handle_glob({"pattern": "*.nomatch"}, workspace_root=workspace)
        == "No files found"
    )
    assert "required" in handle_glob({}, workspace_root=workspace)


def test_glob_cap(workspace: Path) -> None:
    result = handle_glob({"pattern": "**/*"}, workspace_root=workspace, max_results=2)
    assert "Showing 2 of" in result


def test_grep_regex_with_include(workspace: Path) -> None:
    result = handle_grep(
        {"pattern": "needle", "include": "*.py"}, workspace_root=workspace
    )
    assert "Found" in result
    assert "Line 2: needle here" in result
    assert "b.txt" not in result


def test_grep_all_and_empty(workspace: Path) -> None:
    result = handle_grep({"pattern": "needle"}, workspace_root=workspace)
    assert "needle in text" in result
    assert "hidden needle" in result
    assert (
        handle_grep({"pattern": "absent"}, workspace_root=workspace)
        == "No matches found"
    )


def test_grep_validation(workspace: Path) -> None:
    assert "required" in handle_grep({}, workspace_root=workspace)
    assert "invalid regular expression" in handle_grep(
        {"pattern": "["}, workspace_root=workspace
    )
    assert "include" in handle_grep(
        {"pattern": "x", "include": ""}, workspace_root=workspace
    )


@pytest.mark.asyncio
async def test_fs_search_component_registers_both_tools(tmp_path: Path) -> None:
    ctx = Context()
    ctx.use(ToolService)
    ctx.use(SystemPromptService)
    await ctx.flush()
    fs_search_apply(ctx, config={"workspace_root": str(tmp_path)})
    names = {tool["name"] for tool in ctx.tools.schemas()}
    assert "glob" in names
    assert "grep" in names
    (tmp_path / "x.py").write_text("needle here\n", encoding="utf-8")
    g = await ctx.tools.execute(
        ToolCallData(call_id="g1", name="glob", arguments={"pattern": "*.py"})
    )
    assert "x.py" in str(g)
    r = await ctx.tools.execute(
        ToolCallData(call_id="r1", name="grep", arguments={"pattern": "needle"})
    )
    assert "needle" in str(r)


def test_glob_with_path_arg_and_vcs(workspace: Path) -> None:
    # An explicit sub-path anchors the search.
    result = handle_glob({"pattern": "*.py", "path": "src"}, workspace_root=workspace)
    assert "a.py" in result
    # VCS directories are excluded (create one with a matching file).
    vcs = workspace / ".git"
    vcs.mkdir(exist_ok=True)
    (vcs / "secret.py").write_text("x", encoding="utf-8")
    result = handle_glob({"pattern": "**/*.py"}, workspace_root=workspace)
    assert ".git" not in result


def test_glob_invalid_pattern(workspace: Path) -> None:
    result = handle_glob({"pattern": "["}, workspace_root=workspace)
    assert "Error" in result or "No files" in result


def test_grep_line_truncation_and_cap(workspace: Path) -> None:
    (workspace / "long.txt").write_text("needle " + "x" * 3000, encoding="utf-8")
    result = handle_grep(
        {"pattern": "needle"}, workspace_root=workspace, max_line_bytes=50
    )
    assert "(line truncated)" in result
    result = handle_grep({"pattern": "needle"}, workspace_root=workspace, max_matches=1)
    assert "limited by maxMatches" in result


def test_grep_include_with_slash(workspace: Path) -> None:
    result = handle_grep(
        {"pattern": "needle", "include": "src/*.py"}, workspace_root=workspace
    )
    assert "a.py" in result


def test_relative_outside_root(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    assert _relative(workspace, outside) == outside


def test_grep_excludes_vcs(workspace: Path) -> None:
    vcs = workspace / ".git"
    vcs.mkdir(exist_ok=True)
    (vcs / "secret.txt").write_text("needle\n", encoding="utf-8")
    result = handle_grep({"pattern": "needle"}, workspace_root=workspace)
    assert ".git" not in result


def test_apply_without_system_prompt() -> None:
    """The tools mount when no prompt service is present."""
    from animas import Context

    from magitech_harness.tools.fs_search import apply as fs_search_apply
    from magitech_harness.tools.service import ToolsService

    ctx = Context()
    ToolsService(ctx)
    fs_search_apply(ctx)
    assert ctx.tools.get("glob") is not None
    assert ctx.tools.get("grep") is not None
