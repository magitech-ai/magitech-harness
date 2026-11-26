"""Unit tests for bash tool and component."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
from animas import Context

from magitech_harness.shell.service import LocalShellService
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.bash_tool import apply, handle_bash
from magitech_harness.tools.service import ToolService


@pytest.mark.asyncio
async def test_handle_bash_validation(tmp_path: Path) -> None:
    """Verifies argument validation for handle_bash."""
    # Missing command
    res = await handle_bash({})
    assert "required" in res

    # Blank (whitespace-only) command is rejected
    res = await handle_bash({"command": "   "})
    assert "required" in res

    # Nonexistent working directory
    res_bad_cwd = await handle_bash(
        {
            "command": "ls",
            "cwd": str(tmp_path / "nonexistent_directory"),
        }
    )
    assert "Working directory does not exist" in res_bad_cwd


@pytest.mark.asyncio
async def test_handle_bash_success_and_empty(tmp_path: Path) -> None:
    """Verifies successful command execution and empty output case."""
    # Standard output (unquoted: cmd.exe does not strip single quotes)
    res = await handle_bash({"command": "echo Magitech Harness"})
    assert res == "Magitech Harness"

    # Command with explicit valid cwd (python3 prints the native cwd on
    # every platform; Git's pwd would print a POSIX-style path on Windows)
    res_cwd = await handle_bash(
        {
            "command": 'python3 -c "import os; print(os.getcwd())"',
            "cwd": str(tmp_path),
        }
    )
    assert res_cwd == str(tmp_path)

    # Command with no output
    res_empty = await handle_bash({"command": "true"})
    assert res_empty == "(Command executed with no output)"


@pytest.mark.asyncio
async def test_handle_bash_failure() -> None:
    """Verifies handling of non-zero exit codes."""
    # Failure with stderr (&& chains in both sh and cmd.exe; ';' does not)
    res = await handle_bash({"command": "echo an error 1>&2 && exit 1"})
    assert "Command failed with exit code 1:" in res
    assert "an error" in res

    # Failure with stdout
    res_stdout = await handle_bash({"command": "echo stdout fail && exit 2"})
    assert "Command failed with exit code 2:" in res_stdout
    assert "stdout fail" in res_stdout

    # Failure with neither stderr nor stdout
    res_none = await handle_bash({"command": "exit 3"})
    assert "Command failed with exit code 3:" in res_none
    assert "No output produced" in res_none


@pytest.mark.asyncio
async def test_handle_bash_timeout() -> None:
    """Verifies handling when a command times out."""
    res = await handle_bash({"command": "sleep 2", "timeout_ms": 10})
    assert "Command timed out" in res


@pytest.mark.asyncio
async def test_handle_bash_timeout_kill_oserror() -> None:
    """Verifies handling when proc.kill raises OSError during timeout."""
    mock_proc = Mock()
    mock_proc.kill = Mock(side_effect=OSError("Process already dead"))

    async def _mock_timeout(coro, timeout: float) -> None:
        raise TimeoutError()

    with (
        patch("asyncio.create_subprocess_shell", return_value=mock_proc),
        patch("asyncio.wait_for", side_effect=_mock_timeout),
    ):
        res = await handle_bash({"command": "sleep 10", "timeout_ms": 50})
        assert "Command timed out" in res


@pytest.mark.asyncio
async def test_handle_bash_with_ctx_shell_and_spill(tmp_path: Path) -> None:
    """Verifies execution via ctx.shell and automatic spill-to-disk on truncation."""
    ctx = Context()
    ctx.use(LocalShellService)
    await ctx.flush()

    spill_dir = tmp_path / "spill"

    # Small output via ctx.shell
    res_small = await handle_bash(
        {"command": "echo via ctx.shell"},
        ctx=ctx,
        spill_dir=spill_dir,
    )
    assert res_small == "via ctx.shell"
    assert not spill_dir.exists()

    # Large output triggering truncation and spill
    cmd_large = "python3 -c \"for i in range(1000): print(f'line {i}')\""
    res_large = await handle_bash(
        {"command": cmd_large},
        ctx=ctx,
        spill_dir=spill_dir,
        max_output_bytes=200,
    )
    assert "[output truncated; full output:" in res_large
    assert ".stdout" in res_large
    assert spill_dir.exists()

    spill_files = list(spill_dir.glob("*.stdout"))
    assert len(spill_files) == 1
    content = spill_files[0].read_text(encoding="utf-8")
    assert "line 0" in content

    # Test failure via ctx.shell
    res_fail = await handle_bash(
        {"command": "echo fail err 1>&2 && exit 2"},
        ctx=ctx,
        spill_dir=spill_dir,
    )
    assert "Command failed with exit code 2:" in res_fail
    assert "fail err" in res_fail

    # Test timeout via ctx.shell
    res_timeout = await handle_bash(
        {"command": "sleep 5", "timeout_ms": 50},
        ctx=ctx,
        spill_dir=spill_dir,
    )
    assert "Command timed out" in res_timeout


@pytest.mark.asyncio
async def test_handle_bash_spill_oserror_suppressed(tmp_path: Path) -> None:
    """Verifies spill directory creation error is suppressed cleanly."""
    ctx = Context()
    ctx.use(LocalShellService)
    await ctx.flush()

    with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
        res = await handle_bash(
            {"command": "python3 -c \"print('a' * 500)\""},
            ctx=ctx,
            max_output_bytes=100,
        )
        assert "[output truncated; full output:" in res


@pytest.mark.asyncio
async def test_bash_component_integration() -> None:
    """Verifies mounting bash component onto Context."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(LocalShellService)
    ctx.use(ToolService)
    ctx.use(apply)
    await ctx.flush()

    assert ctx.tools.get("bash") is not None
    tool = ctx.tools.get("bash")
    assert tool is not None
    assert tool["name"] == "bash"

    res = await tool["execute"]({"command": "echo registered bash tool"})
    assert res == "registered bash tool"


@pytest.mark.asyncio
async def test_bash_component_registers_system_prompt_section() -> None:
    """Verifies the component registers tool:bash prompt guidance."""
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(LocalShellService)
    ctx.use(ToolService)
    ctx.use(apply)
    await ctx.flush()

    assert ctx.systemPrompt.get_section_order("TOOL_BASH") == 1000


@pytest.mark.asyncio
async def test_bash_component_without_system_prompt_service() -> None:
    """Verifies the component mounts and skips section when systemPrompt absent."""
    fake_tools = type(
        "FakeTools",
        (),
        {"register": lambda self, tool: None},
    )()
    ctx = Context()
    ctx.reflect.provide("tools", fake_tools, None)
    await ctx.flush()

    ctx.use(LocalShellService)
    await ctx.flush()
    ctx.use(apply)
    await ctx.flush()

    assert ctx.has("tools")


@pytest.mark.asyncio
async def test_bash_escalation_paths(tmp_path: Path) -> None:
    from magitech_harness.tools.bash_tool import handle_bash

    class Approver:
        def __init__(self, status: str = "allowed-once") -> None:
            self.status = status

        async def request(self, _payload: dict[str, Any]) -> dict[str, str]:
            return {"status": self.status}

    class Shell:
        async def run(self, command: str, **kwargs: Any) -> Any:
            return SimpleNamespace(
                exit_code=0,
                stdout="ok",
                stderr="",
                timed_out=False,
                truncated=False,
                stdout_spill_path=None,
                stderr_spill_path=None,
                sandbox={"mode": "read-only", "denied": False},
            )

    def make_ctx(approver: Approver) -> Context:
        ctx = Context()
        ctx.provide(
            "sandboxPolicy",
            SimpleNamespace(
                default_mode="read-only",
                resolve=lambda session: {
                    "mode": "read-only",
                    "workspaceRoot": str(tmp_path),
                },
            ),
        )
        ctx.provide("approval", approver)
        ctx.provide("shell", Shell())
        return ctx

    ctx = make_ctx(Approver())
    view = {
        "agent": SimpleNamespace(session=SimpleNamespace()),
        "callId": "c1",
        "signal": None,
    }
    out = await handle_bash(
        {
            "command": "echo ok",
            "description": "echo",
            "sandbox_permissions": "danger-full-access",
            "justification": "need full access",
        },
        ctx=ctx,
        exec_view=view,
    )
    assert "ok" in out

    assert "sandbox_permissions" in await handle_bash(
        {"command": "echo ok", "sandbox_permissions": "workspace-write"},
        ctx=ctx,
        exec_view=view,
    )
    assert "invalid sandbox_permissions" in await handle_bash(
        {"command": "echo ok", "sandbox_permissions": "bogus", "justification": "why"},
        ctx=ctx,
        exec_view=view,
    )
    rejected = make_ctx(Approver("rejected"))
    assert "Error" in await handle_bash(
        {
            "command": "echo ok",
            "sandbox_permissions": "danger-full-access",
            "justification": "why",
        },
        ctx=rejected,
        exec_view=view,
    )


@pytest.mark.asyncio
async def test_bash_denial_marker(tmp_path: Path) -> None:
    from magitech_harness.tools.bash_tool import handle_bash

    class DenyingShell:
        async def run(self, command: str, **kwargs: Any) -> Any:
            return SimpleNamespace(
                exit_code=1,
                stdout="",
                stderr="read-only file system",
                timed_out=False,
                truncated=False,
                stdout_spill_path=None,
                stderr_spill_path=None,
                sandbox={"mode": "read-only", "denied": True, "enforcement": "full"},
            )

    ctx = Context()
    ctx.provide("shell", DenyingShell())
    out = await handle_bash({"command": "touch /etc/x"}, ctx=ctx)
    assert "[sandbox: file access denied under read-only mode]" in out
    assert "escalation available" in out


@pytest.mark.asyncio
async def test_bash_under_policy_without_escalation(tmp_path: Path) -> None:
    from magitech_harness.tools.bash_tool import handle_bash

    class Shell:
        async def run(self, command: str, **kwargs: Any) -> Any:
            assert "sandboxPolicy" in kwargs
            return SimpleNamespace(
                exit_code=0,
                stdout="fine",
                stderr="",
                timed_out=False,
                truncated=False,
                stdout_spill_path=None,
                stderr_spill_path=None,
                sandbox=None,
            )

    ctx = Context()
    ctx.provide(
        "sandboxPolicy",
        SimpleNamespace(
            default_mode="workspace-write",
            resolve=lambda session: {
                "mode": "workspace-write",
                "workspaceRoot": str(tmp_path),
            },
        ),
    )
    ctx.provide("shell", Shell())
    assert "fine" in await handle_bash({"command": "echo fine"}, ctx=ctx)


@pytest.mark.asyncio
async def test_executor_sandbox_policy_kwarg(tmp_path: Path) -> None:
    from magitech_harness import create_harness

    ctx = create_harness(storage_dir=tmp_path / "executor-policy")
    await ctx.flush()
    executor = ctx.get("shell", None, strict=False)
    result = await executor.run(
        "echo policy",
        sandboxPolicy={"mode": "danger-full-access", "workspaceRoot": str(tmp_path)},
    )
    assert result.stdout.strip() == "policy"
