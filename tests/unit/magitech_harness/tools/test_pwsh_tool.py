"""Tests for the model-facing PowerShell tool."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness.tools.pwsh_tool import (
    PWSH_DESCRIPTION,
    PWSH_SECTION,
    apply,
    handle_pwsh,
)

PWSH_BIN = shutil.which("pwsh") or "/usr/bin/pwsh"


@pytest.mark.asyncio
async def test_pwsh_execution(tmp_path: Any) -> None:
    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    result = await handle_pwsh(
        {"description": "add", "command": "20+22"}, ctx=ctx, spill_dir=tmp_path
    )
    assert "42" in result

    result = await handle_pwsh(
        {"command": "Write-Output 'hi'; Write-Error 'bad'"}, ctx=ctx, spill_dir=tmp_path
    )
    assert "hi" in result and "[stderr]" in result

    result = await handle_pwsh({"command": "exit 7"}, ctx=ctx, spill_dir=tmp_path)
    assert "[exit code: 7]" in result

    workdir = tmp_path / "work"
    workdir.mkdir()
    result = await handle_pwsh(
        {"command": "(Get-Location).Path", "workdir": str(workdir)},
        ctx=ctx,
        spill_dir=tmp_path,
    )
    assert str(workdir) in result


@pytest.mark.asyncio
async def test_pwsh_validation_and_errors(tmp_path: Any) -> None:
    assert "command" in await handle_pwsh({})
    assert "command" in await handle_pwsh({"command": "   "})
    assert "timeoutMs" in await handle_pwsh({"command": "1", "timeoutMs": "x"})
    assert "Working directory" in await handle_pwsh(
        {"command": "1", "workdir": str(tmp_path / "missing")}
    )
    assert "1" in await handle_pwsh({"command": "1"}, ctx=None)

    old = os.environ.get("PATH", "")
    try:
        os.environ["PATH"] = "/nonexistent"
        result = await handle_pwsh({"command": "1"}, ctx=None)
        assert "could not start PowerShell" in result
    finally:
        os.environ["PATH"] = old


@pytest.mark.asyncio
async def test_pwsh_timeout_and_spill(tmp_path: Any) -> None:
    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    result = await handle_pwsh(
        {"command": "Start-Sleep -Seconds 2", "timeoutMs": 50},
        ctx=ctx,
        spill_dir=tmp_path,
    )
    assert "timed out" in result

    result = await handle_pwsh(
        {"command": "'x' * 100000", "command_timeout": 1},
        ctx=ctx,
        spill_dir=tmp_path,
        max_output_bytes=10,
    )
    assert "truncated" in result


@pytest.mark.asyncio
async def test_pwsh_apply() -> None:

    ctx = Context()
    registered: list[Any] = []
    ctx.provide("tools", SimpleNamespace(register=registered.append))
    ctx.provide("pwshConfig", PWSH_BIN)
    apply(ctx)
    assert registered[0]["name"] == "pwsh"
    assert "42" in await registered[0]["execute"](
        {"description": "add", "command": "20+22"}
    )
    assert PWSH_DESCRIPTION and PWSH_SECTION


@pytest.mark.asyncio
async def test_pwsh_spill_failure_and_prompt(monkeypatch: Any, tmp_path: Any) -> None:

    ctx = Context()
    ctx.provide("pwshConfig", PWSH_BIN)
    original = Path.write_bytes

    def fail_write(self: Path, data: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    result = await handle_pwsh(
        {"command": "'x' * 100000"},
        ctx=ctx,
        spill_dir=tmp_path,
        max_output_bytes=10,
    )
    assert "unavailable" in result
    monkeypatch.setattr(Path, "write_bytes", original)

    registered: list[Any] = []
    sections: list[Any] = []
    ctx2 = Context()
    ctx2.provide("tools", SimpleNamespace(register=registered.append))
    ctx2.provide(
        "systemPrompt",
        SimpleNamespace(
            section=sections.append,
            get_section_order=lambda name: 1000,
        ),
    )
    apply(ctx2)
    assert sections[0]["name"] == "tool:pwsh"
