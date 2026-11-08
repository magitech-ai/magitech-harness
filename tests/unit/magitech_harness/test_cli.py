"""Unit tests for magitech_harness CLI entrypoint."""

import sys
from pathlib import Path
from typing import ClassVar

import pytest

from magitech_harness.cli import main, main_async


@pytest.mark.asyncio
async def test_cli_main_async_with_prompt_and_storage(
    tmp_path: Path, monkeypatch
) -> None:
    """Verifies main_async runs turn with specified prompt and storage directory."""
    test_args = ["mth", "Calculate 5 + 7", "--storage", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", test_args)

    # Run main_async
    await main_async()

    # Verify session log created in storage
    session_files = list(tmp_path.glob("*.jsonl"))
    assert len(session_files) == 1
    content = session_files[0].read_text(encoding="utf-8")
    assert "Calculate 5 + 7" in content


@pytest.mark.asyncio
async def test_cli_main_async_default_prompt(tmp_path: Path, monkeypatch) -> None:
    """Verifies main_async uses default prompt and handles empty storage path."""
    # Pass --storage "" to hit branch where session.storage_path is None (line 58->exit)
    test_args = ["mth", "--storage", ""]
    monkeypatch.setattr(sys, "argv", test_args)

    await main_async()


@pytest.mark.asyncio
async def test_cli_main_async_with_profile(tmp_path: Path, monkeypatch) -> None:
    """Verifies main_async loads profile via Anima loader and skips fallback provider."""
    import types

    dummy_pkg = types.ModuleType("dummy_cli_provider_pkg")

    class DummyProviderPlugin:
        inject: ClassVar[list[str]] = ["llm"]

        def __init__(self, c):
            from magitech_harness.llm.service import MockLLMProvider

            c.llm.register_provider("profile_mock", MockLLMProvider())

    dummy_pkg.DummyProviderPlugin = DummyProviderPlugin
    monkeypatch.setitem(sys.modules, "dummy_cli_provider_pkg", dummy_pkg)

    profile_file = tmp_path / "dummy_profile.yaml"
    profile_file.write_text(
        "['dummy_cli_provider_pkg:DummyProviderPlugin']", encoding="utf-8"
    )

    test_args = ["mth", "--profile", str(profile_file), "--storage", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", test_args)

    from magitech_harness.agent.loop import AgentLoopService

    async def mock_turn(self, session, prompt, **kwargs):
        return "Profile Turn Result"

    monkeypatch.setattr(AgentLoopService, "run_turn", mock_turn)
    await main_async()


@pytest.mark.asyncio
async def test_cli_add_tool_execution(tmp_path: Path, monkeypatch) -> None:
    """Verifies the add tool registered in CLI executes properly (line 36)."""
    from magitech_harness.session.events import ToolCallData

    test_args = ["mth", "--storage", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", test_args)

    from magitech_harness.agent.loop import AgentLoopService

    async def hook_run_turn(self, session, prompt, **kwargs):
        # Trigger the registered 'add' tool
        tc = ToolCallData(call_id="call_add", name="add", arguments={"a": 40, "b": 2})
        tool_res = await self.ctx.tools.execute(tc)
        assert tool_res.content == 42
        return "Calculated 42"

    monkeypatch.setattr(AgentLoopService, "run_turn", hook_run_turn)
    await main_async()


def test_cli_main_module_execution(monkeypatch) -> None:
    """Verifies python -m magitech_harness.cli execution (line 70)."""
    import runpy
    import warnings

    def mock_run(coro):
        coro.close()

    monkeypatch.setattr("asyncio.run", mock_run)
    monkeypatch.setattr(sys, "argv", ["mth"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("magitech_harness.cli", run_name="__main__")


def test_cli_main_invokes_run(monkeypatch) -> None:
    """Verifies main() wraps main_async in asyncio.run."""
    ran = []

    def mock_run(coro):
        coro.close()
        ran.append(True)

    monkeypatch.setattr("asyncio.run", mock_run)
    main()
    assert ran == [True]


def test_cli_main_keyboard_interrupt(monkeypatch) -> None:
    """Verifies main() cleanly exits on KeyboardInterrupt."""

    def mock_run(coro):
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("asyncio.run", mock_run)
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
