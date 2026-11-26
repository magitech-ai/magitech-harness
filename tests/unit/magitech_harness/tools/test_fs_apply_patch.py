"""Tests for the config-gated ``apply_patch`` tool."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness.tools import fs_apply_patch as patch_tool_module
from magitech_harness.tools.fs_apply_patch import (
    APPLY_PATCH_SECTION_TEXT,
    handle_apply_patch,
    register_apply_patch,
)


def _patch(*body: str) -> str:
    return "\n".join(["*** Begin Patch", *body, "*** End Patch"])


@pytest.mark.asyncio
async def test_apply_patch_full_lifecycle(tmp_path: Any) -> None:
    (tmp_path / "old.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "move.txt").write_text("ctx\na\n", encoding="utf-8")
    (tmp_path / "gone.txt").write_text("bye\n", encoding="utf-8")
    raw = _patch(
        "*** Add File: new.txt",
        "+hello",
        "*** Update File: old.txt",
        "@@",
        "-old",
        "+new",
        "*** Update File: move.txt",
        "*** Move to: moved.txt",
        "@@ ctx",
        "-a",
        "+b",
        "*** Delete File: gone.txt",
    )
    result = await handle_apply_patch({"input": raw}, workspace_root=tmp_path)
    assert result == (
        "Success. Updated the following files:\n"
        "A new.txt\nM old.txt\nM moved.txt\nD gone.txt"
    )
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\n"
    assert (tmp_path / "old.txt").read_text(encoding="utf-8") == "new\n"
    assert (tmp_path / "moved.txt").read_text(encoding="utf-8") == "ctx\nb\n"
    assert not (tmp_path / "move.txt").exists()
    assert not (tmp_path / "gone.txt").exists()


@pytest.mark.asyncio
async def test_apply_patch_no_operations_and_validation(tmp_path: Any) -> None:
    assert await handle_apply_patch({"input": _patch()}) == (
        "Success. No operations in patch."
    )
    assert await handle_apply_patch({}) == "Error: 'input' argument is required."
    assert await handle_apply_patch({"input": 42}) == (
        "Error: 'input' argument is required."
    )
    assert "Invalid patch at line" in await handle_apply_patch({"input": "not a patch"})


@pytest.mark.asyncio
async def test_apply_patch_preflight_refusals(tmp_path: Any) -> None:
    # Hunk mismatch is reported before any mutation.
    (tmp_path / "keep.txt").write_text("unchanged\n", encoding="utf-8")
    mismatch = await handle_apply_patch(
        {"input": _patch("*** Update File: keep.txt", "@@", "-missing", "+other")},
        workspace_root=tmp_path,
    )
    assert mismatch.startswith("Error: Patch rejected before any change (1 problems):")
    assert "does not match" in mismatch
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "unchanged\n"

    for kind, header in (
        ("update", "*** Update File: absent.txt"),
        ("delete", "*** Delete File: absent.txt"),
        ("move", "*** Update File: absent.txt"),
    ):
        body = [header]
        if kind == "move":
            body.append("*** Move to: target.txt")
        if kind in ("update", "move"):
            body.extend(["@@", "-x", "+y"])
        reason = await handle_apply_patch(
            {"input": _patch(*body)}, workspace_root=tmp_path
        )
        assert f'cannot {kind} "absent.txt": not found' in reason

    (tmp_path / "self.txt").write_text("x\n", encoding="utf-8")
    self_move = await handle_apply_patch(
        {
            "input": _patch(
                "*** Update File: self.txt",
                "*** Move to: self.txt",
                "@@",
                "-x",
                "+y",
            )
        },
        workspace_root=tmp_path,
    )
    assert 'cannot move "self.txt" to itself' in self_move

    (tmp_path / "dir").mkdir()
    add_dir = await handle_apply_patch(
        {"input": _patch("*** Add File: dir", "+x")}, workspace_root=tmp_path
    )
    assert 'cannot add "dir": not a regular file' in add_dir


@pytest.mark.asyncio
async def test_apply_patch_binary_and_long_preview(tmp_path: Any) -> None:
    (tmp_path / "binary.txt").write_bytes(b"a\0b\n")
    binary = await handle_apply_patch(
        {"input": _patch("*** Update File: binary.txt", "@@", "-x", "+y")},
        workspace_root=tmp_path,
    )
    assert 'cannot patch "binary.txt": binary file' in binary

    long_line = "x" * 100
    (tmp_path / "long.txt").write_text(long_line + "\n", encoding="utf-8")
    preview = await handle_apply_patch(
        {"input": _patch("*** Update File: long.txt", "@@", f"-{long_line}z", "+fin")},
        workspace_root=tmp_path,
    )
    assert "…" in preview


@pytest.mark.asyncio
async def test_apply_patch_symlink_source(tmp_path: Any) -> None:
    target = tmp_path / "target.txt"
    target.write_text("x\n", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    for kind, body in (
        ("delete", ["*** Delete File: link.txt"]),
        (
            "move",
            ["*** Update File: link.txt", "*** Move to: other.txt", "@@", "-x", "+y"],
        ),
    ):
        reason = await handle_apply_patch(
            {"input": _patch(*body)}, workspace_root=tmp_path
        )
        assert f'cannot {kind} "link.txt": path is a symbolic link' in reason


@pytest.mark.asyncio
async def test_apply_patch_sandbox_and_workspace_denials(tmp_path: Any) -> None:
    ctx = Context()
    ctx.provide(
        "sandboxPolicy",
        SimpleNamespace(
            resolve=lambda _session: {
                "mode": "read-only",
                "workspaceRoot": str(tmp_path),
            }
        ),
    )
    denied = await handle_apply_patch(
        {"input": _patch("*** Add File: blocked.txt", "+x")}, ctx=ctx
    )
    assert "[sandbox: file access denied under read-only mode]" in denied

    escaped = await handle_apply_patch(
        {"input": _patch("*** Add File: ../escape.txt", "+x")},
        workspace_root=tmp_path / "nested",
    )
    assert escaped.startswith("Error: Patch rejected before any change")


@pytest.mark.asyncio
async def test_apply_patch_commit_failure_keeps_earlier_changes(
    monkeypatch: Any, tmp_path: Any
) -> None:
    original = Path.write_text

    def fail_blocked(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.name == "blocked.txt":
            raise OSError("disk full")
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_blocked)
    result = await handle_apply_patch(
        {
            "input": _patch(
                "*** Add File: first.txt",
                "+ok",
                "*** Add File: blocked.txt",
                "+no",
            )
        },
        workspace_root=tmp_path,
    )
    assert result.startswith("Patch stopped after 1 of 2 operations.")
    assert "Completed:\nA first.txt" in result
    assert "Failed:\n- operation 2 (add blocked.txt" in result
    assert "disk full" in result
    assert (tmp_path / "first.txt").exists()
    assert not (tmp_path / "blocked.txt").exists()

    original_unlink = Path.unlink

    def fail_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        raise OSError("busy")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    (tmp_path / "stubborn.txt").write_text("x\n", encoding="utf-8")
    deleted = await handle_apply_patch(
        {"input": _patch("*** Delete File: stubborn.txt")},
        workspace_root=tmp_path,
    )
    assert deleted.startswith("Error: Patch rejected before any change (1 problems):")
    assert "cannot delete" in deleted
    monkeypatch.setattr(Path, "unlink", original_unlink)


@pytest.mark.asyncio
async def test_apply_patch_read_failure(monkeypatch: Any, tmp_path: Any) -> None:
    target = tmp_path / "unreadable.txt"
    target.write_text("x\n", encoding="utf-8")
    original = Path.read_text

    def fail_read(self: Path, *args: Any, **kwargs: Any) -> str:
        if self == target:
            raise OSError("gone")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read)
    reason = await handle_apply_patch(
        {"input": _patch("*** Update File: unreadable.txt", "@@", "-x", "+y")},
        workspace_root=tmp_path,
    )
    assert "cannot read" in reason


@pytest.mark.asyncio
async def test_register_apply_patch_guidance() -> None:
    ctx = Context()
    sections: list[Any] = []
    ctx.provide(
        "tools",
        SimpleNamespace(register=lambda definition: sections.append(definition)),
    )
    ctx.provide(
        "systemPrompt",
        SimpleNamespace(
            section=lambda section: sections.append(section),
            get_section_order=lambda _name: 1350,
        ),
    )
    register_apply_patch(ctx)
    assert sections[0]["name"] == "apply_patch"
    assert (
        "*** Begin Patch"
        in sections[0]["parameters"]["properties"]["input"]["description"]
    )
    assert sections[1]["name"] == "tool:apply_patch"
    assert sections[1]["text"] == APPLY_PATCH_SECTION_TEXT
    assert sections[1]["order"] == 1350


@pytest.mark.asyncio
async def test_apply_patch_reuses_state_and_content(tmp_path: Any) -> None:
    target = tmp_path / "twice.txt"
    target.write_text("one\n", encoding="utf-8")
    result = await handle_apply_patch(
        {
            "input": _patch(
                "*** Update File: twice.txt",
                "@@",
                "-one",
                "+two",
                "*** Update File: twice.txt",
                "@@",
                "-two",
                "+three",
            )
        },
        workspace_root=tmp_path,
    )
    assert result == ("Success. Updated the following files:\nM twice.txt\nM twice.txt")
    assert target.read_text(encoding="utf-8") == "three\n"


@pytest.mark.asyncio
async def test_apply_patch_non_regular_and_destination_denials(tmp_path: Any) -> None:
    directory = tmp_path / "dir"
    directory.mkdir()
    deleted = await handle_apply_patch(
        {"input": _patch("*** Delete File: dir")}, workspace_root=tmp_path
    )
    assert 'cannot delete "dir": not a regular file' in deleted

    updated = await handle_apply_patch(
        {"input": _patch("*** Update File: dir", "@@", "-x", "+y")},
        workspace_root=tmp_path,
    )
    assert 'cannot update "dir": not a regular file' in updated

    (tmp_path / "src.txt").write_text("x\n", encoding="utf-8")
    moved = await handle_apply_patch(
        {
            "input": _patch(
                "*** Update File: src.txt",
                "*** Move to: dir",
                "@@",
                "-x",
                "+y",
            )
        },
        workspace_root=tmp_path,
    )
    assert 'cannot move "dir": not a regular file' in moved


@pytest.mark.asyncio
async def test_apply_patch_skips_successors_after_failure(
    monkeypatch: Any, tmp_path: Any
) -> None:
    original = Path.write_text

    def fail_blocked(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if self.name == "blocked.txt":
            raise OSError("disk full")
        return original(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_blocked)
    result = await handle_apply_patch(
        {
            "input": _patch(
                "*** Add File: first.txt",
                "+1",
                "*** Add File: blocked.txt",
                "+2",
                "*** Add File: third.txt",
                "+3",
            )
        },
        workspace_root=tmp_path,
    )
    assert result.startswith("Patch stopped after 1 of 3 operations.")
    assert "Not executed:\n- operation 3 (add third.txt)" in result


@pytest.mark.asyncio
async def test_register_apply_patch_execute_and_no_prompt() -> None:
    ctx = Context()
    registered: list[Any] = []
    ctx.provide("tools", SimpleNamespace(register=registered.append))
    register_apply_patch(ctx)
    assert await registered[0]["execute"]({"input": _patch()}) == (
        "Success. No operations in patch."
    )

    bare = Context()
    bare.provide("tools", SimpleNamespace(register=lambda _definition: None))
    register_apply_patch(bare)


@pytest.mark.asyncio
async def test_apply_patch_destination_denial(monkeypatch: Any, tmp_path: Any) -> None:
    (tmp_path / "src.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.setattr(
        patch_tool_module,
        "_denied",
        lambda _ctx, _view, path_str, _resolved=None: (
            "[sandbox: file access denied under read-only mode]"
            if path_str == "outside.txt"
            else None
        ),
    )
    reason = await handle_apply_patch(
        {
            "input": _patch(
                "*** Update File: src.txt",
                "*** Move to: outside.txt",
                "@@",
                "-x",
                "+y",
            )
        },
        workspace_root=tmp_path,
    )
    assert "outside.txt" in reason
    assert "file access denied" in reason
    assert (tmp_path / "src.txt").read_text(encoding="utf-8") == "x\n"


@pytest.mark.asyncio
async def test_apply_patch_detects_concurrent_change(
    monkeypatch: Any, tmp_path: Any
) -> None:
    target = tmp_path / "racy.txt"
    target.write_text("old\n", encoding="utf-8")
    original = Path.read_text
    calls = 0

    def changing(self: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal calls
        text = original(self, *args, **kwargs)
        calls += 1
        if calls == 1:
            self.write_text("concurrent\n", encoding="utf-8")
        return text

    monkeypatch.setattr(Path, "read_text", changing)
    result = await handle_apply_patch(
        {"input": _patch("*** Update File: racy.txt", "@@", "-old", "+new")},
        workspace_root=tmp_path,
    )
    assert "file changed since it was read" in result
    assert target.read_text(encoding="utf-8") == "concurrent\n"


@pytest.mark.asyncio
async def test_apply_patch_invalid_utf8_is_refused(tmp_path: Any) -> None:
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\n")
    reason = await handle_apply_patch(
        {"input": _patch("*** Update File: bad.txt", "@@", "-x", "+y")},
        workspace_root=tmp_path,
    )
    assert 'cannot patch "bad.txt": binary file' in reason


@pytest.mark.asyncio
async def test_apply_patch_groups_success_lines(tmp_path: Any) -> None:
    (tmp_path / "edit.txt").write_text("old\n", encoding="utf-8")
    (tmp_path / "delete.txt").write_text("gone\n", encoding="utf-8")
    (tmp_path / "move-src.txt").write_text("x\n", encoding="utf-8")
    result = await handle_apply_patch(
        {
            "input": _patch(
                "*** Update File: edit.txt",
                "@@",
                "-old",
                "+new",
                "*** Add File: created.txt",
                "+hello",
                "*** Delete File: delete.txt",
                "*** Update File: move-src.txt",
                "*** Move to: move-dst.txt",
                "@@",
                "-x",
                "+y",
            )
        },
        workspace_root=tmp_path,
    )
    assert result == (
        "Success. Updated the following files:\n"
        "A created.txt\nM edit.txt\nM move-dst.txt\nD delete.txt"
    )


@pytest.mark.asyncio
async def test_apply_patch_partial_move_diagnostic(
    monkeypatch: Any, tmp_path: Any
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("x\n", encoding="utf-8")
    original = Path.unlink

    def fail_source(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == source:
            raise OSError("busy")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source)
    result = await handle_apply_patch(
        {
            "input": _patch(
                "*** Update File: source.txt",
                "*** Move to: destination.txt",
                "@@",
                "-x",
                "+y",
            )
        },
        workspace_root=tmp_path,
    )
    assert "Patch stopped after 0 of 1 operations." in result
    assert (
        "destination destination.txt was written; source source.txt remains" in result
    )
    assert (tmp_path / "destination.txt").read_text(encoding="utf-8") == "y\n"


@pytest.mark.asyncio
async def test_apply_patch_commit_hunk_failure(monkeypatch: Any, tmp_path: Any) -> None:
    (tmp_path / "shifting.txt").write_text("old\n", encoding="utf-8")
    original = patch_tool_module.apply_hunks
    calls = 0

    def shifting(before: str, hunks: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls > 1:
            return {
                "ok": False,
                "failure": {
                    "hunkLineNumber": 2,
                    "stage": "oldLines",
                    "searchStart": 0,
                    "pattern": ["old"],
                },
            }
        return original(before, hunks)

    monkeypatch.setattr(patch_tool_module, "apply_hunks", shifting)
    result = await handle_apply_patch(
        {"input": _patch("*** Update File: shifting.txt", "@@", "-old", "+new")},
        workspace_root=tmp_path,
    )
    assert "hunk no longer matches" in result


@pytest.mark.asyncio
async def test_apply_patch_commit_binary_read(monkeypatch: Any, tmp_path: Any) -> None:
    target = tmp_path / "flips.txt"
    target.write_text("old\n", encoding="utf-8")
    original = Path.read_text
    calls = 0

    def flips(self: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flips)
    result = await handle_apply_patch(
        {"input": _patch("*** Update File: flips.txt", "@@", "-old", "+new")},
        workspace_root=tmp_path,
    )
    assert 'cannot patch "flips.txt": binary file' in result
