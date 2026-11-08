"""Unit tests for LocalShellService."""

import pytest
from anima import Context
from magitech_harness.shell.service import LocalShellService
from magitech_harness.tools.service import ToolService


@pytest.mark.asyncio
async def test_shell_service_run_and_bash_tool() -> None:
    """Verifies shell execution and bash tool formatting."""
    ctx = Context()
    ctx.plugin(ToolService)
    ctx.plugin(LocalShellService)
    await ctx.flush()

    assert ctx.has("shell")
    assert ctx.tools.get("bash") is not None

    # Execute bash tool directly
    res = await ctx.shell._execute_bash_tool({"command": "echo 'hello stdout' && echo 'error stderr' >&2"})
    assert "hello stdout" in res
    assert "STDERR:\nerror stderr" in res
    assert "[Exit code: 0]" in res

    # Command with no stdout and no stderr (branches 42->44 and 44->46)
    empty_res = await ctx.shell._execute_bash_tool({"command": "true"})
    assert empty_res == "[Exit code: 0]"


@pytest.mark.asyncio
async def test_shell_service_without_tools() -> None:
    """Verifies LocalShellService operates when ToolService is absent."""
    ctx = Context()
    service = LocalShellService(ctx)
    code, stdout, stderr = await service.run("echo test")
    assert code == 0
    assert "test" in stdout
