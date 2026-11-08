"""Unit tests for LocalShellService."""

from pathlib import Path

import pytest
from anima import Context

from magitech_harness.shell.service import LocalShellService
from magitech_harness.tools.service import ToolService


@pytest.mark.asyncio
async def test_shell_service_run(tmp_path: Path) -> None:
    """Verifies shell execution returns code, stdout, and stderr, respecting cwd."""
    ctx = Context()
    ctx.plugin(LocalShellService)
    await ctx.flush()

    assert ctx.has("shell")

    # Command with both stdout and stderr
    code, stdout, stderr = await ctx.shell.run(
        "echo 'hello stdout' && echo 'error stderr' >&2"
    )
    assert code == 0
    assert "hello stdout" in stdout
    assert "error stderr" in stderr

    # Command respecting cwd
    code, stdout, _stderr = await ctx.shell.run("pwd", cwd=tmp_path)
    assert code == 0
    assert str(tmp_path.resolve()) in stdout

    # Failing command
    code, _stdout, _stderr = await ctx.shell.run("false")
    assert code != 0


@pytest.mark.asyncio
async def test_shell_service_does_not_register_tools() -> None:
    """Verifies LocalShellService strictly acts as an execution backend with zero domain tools."""
    ctx = Context()
    ctx.plugin(ToolService)
    ctx.plugin(LocalShellService)
    await ctx.flush()

    assert ctx.has("shell")
    assert ctx.has("tools")
    # Strictly zero built-in domain tools in core harness
    assert ctx.tools.get("bash") is None
