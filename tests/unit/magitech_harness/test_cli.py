"""Unit tests for magitech_harness CLI entrypoint and interactive REPL."""

import asyncio
import sys
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from anima import Context, EventMode

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortError
from magitech_harness.cli import (
    ReplDispatcher,
    main,
    main_async,
    run_repl,
)
from magitech_harness.llm.service import MockLLMProvider
from magitech_harness.session.events import EventType, ToolCallData


@pytest.mark.asyncio
async def test_repl_dispatcher_all_commands(tmp_path: Path) -> None:
    """Verifies ReplDispatcher handling of all slash commands."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()

    s = ctx.sessions.create("sess_main")
    ctx.sessions.log_event(s, EventType.USER_MESSAGE, {"text": "persisted"})

    out_lines = []

    def capture_out(*args, **kwargs) -> None:
        out_lines.append(" ".join(str(a) for a in args))

    dispatcher = ReplDispatcher(ctx, s, model="default-model", out=capture_out)

    # /help
    assert await dispatcher.dispatch("/help") is False
    assert any("Available slash commands" in line for line in out_lines)

    # /model without arg
    assert await dispatcher.dispatch("/model") is False
    assert any("Current model: default-model" in line for line in out_lines)

    # /model with arg
    assert await dispatcher.dispatch("/model custom/model-x") is False
    assert dispatcher.model == "custom/model-x"
    assert any("Active model set to: custom/model-x" in line for line in out_lines)

    # /clear
    assert await dispatcher.dispatch("/clear") is False

    # /session list
    assert await dispatcher.dispatch("/session list") is False
    assert any("Sessions" in line for line in out_lines)

    # /session new
    assert await dispatcher.dispatch("/session new") is False
    assert dispatcher.session.session_id != "sess_main"

    # /session show on current session
    assert await dispatcher.dispatch("/session show") is False

    # /session show on specific existing session
    assert await dispatcher.dispatch("/session show sess_main") is False
    assert any("persisted" in line for line in out_lines)

    # /session show on missing session
    assert await dispatcher.dispatch("/session show nonexistent") is False
    assert any("Session not found: nonexistent" in line for line in out_lines)

    # /session unknown subcommand
    assert await dispatcher.dispatch("/session invalid_sub") is False
    assert any("Unknown session subcommand: invalid_sub" in line for line in out_lines)

    # /compact with default handler (no range)
    assert await dispatcher.dispatch("/compact") is False
    assert any("no compactable range found" in line for line in out_lines)

    # /compact when compaction component reports compacted
    async def mock_compact(payload, next_fn):
        return {"compacted": True, "summary": "Folded 5 events"}

    ctx.on("session/compact", mock_compact, mode=EventMode.WATERFALL)
    assert await dispatcher.dispatch("/compact") is False
    assert any("Folded 5 events" in line for line in out_lines)

    # Unknown command
    assert await dispatcher.dispatch("/foobar") is False
    assert any("Unknown command: /foobar" in line for line in out_lines)

    # /exit and /quit
    assert await dispatcher.dispatch("/exit") is True
    assert await dispatcher.dispatch("/quit") is True


@pytest.mark.asyncio
async def test_run_repl_interactive_stream_and_tools() -> None:
    """Verifies run_repl prompt loop with streamed chunks and tool events."""
    ctx = create_harness(mock_provider=True)
    await ctx.flush()

    mock_llm: MockLLMProvider = ctx.llm.active_provider
    tc = ToolCallData(call_id="c1", name="dummy_tool", arguments={"val": 1})
    tc_err = ToolCallData(call_id="c2", name="error_tool", arguments={})
    mock_llm.queue_response(text="Tool result needed", tool_calls=[tc, tc_err])
    mock_llm.queue_response(text="Final answer")

    # Register dummy tools
    from magitech_harness.tools.definition import define_tool

    ctx.tools.register(
        define_tool(
            name="dummy_tool",
            description="dummy",
            execute=lambda args: "dummy success",
        )
    )
    ctx.tools.register(
        define_tool(
            name="error_tool",
            description="fails",
            execute=lambda args: (_ for _ in ()).throw(ValueError("boom")),
        )
    )

    inputs = ["", "/help", "Run tool query", "interrupted_after"]
    input_idx = 0

    def mock_input() -> str:
        nonlocal input_idx
        if input_idx < len(inputs):
            res = inputs[input_idx]
            input_idx += 1
            if res == "interrupted_after":
                raise KeyboardInterrupt
            return res
        raise EOFError

    out_lines = []

    def mock_out(*args, **kwargs) -> None:
        out_lines.append(" ".join(str(a) for a in args))

    await run_repl(ctx, input_fn=mock_input, out_fn=mock_out)

    assert any("Tool Start" in line for line in out_lines)
    assert any("Tool Complete" in line or "OK" in line for line in out_lines)
    assert any("Final answer" in line for line in out_lines)


@pytest.mark.asyncio
async def test_run_repl_signal_handler_unsupported_and_usage_event() -> None:
    """Verifies run_repl when loop.add_signal_handler raises NotImplementedError and handles usage events."""
    ctx = create_harness(mock_provider=True)
    await ctx.flush()

    async def mock_stream_turn(session, prompt, **kwargs):
        yield {"type": "usage", "usage": {"total_tokens": 100}}
        yield {"type": "chunk", "text": "streamed reply"}

    ctx.agent_loop.stream_turn = mock_stream_turn

    inputs = ["hello", "/exit"]
    input_idx = 0

    def mock_input() -> str:
        nonlocal input_idx
        res = inputs[input_idx]
        input_idx += 1
        return res

    out_lines = []

    def mock_out(*args, **kwargs) -> None:
        out_lines.append(" ".join(str(a) for a in args))

    # Mock loop.add_signal_handler to raise NotImplementedError
    loop = asyncio.get_running_loop()
    with patch.object(loop, "add_signal_handler", side_effect=NotImplementedError):
        await run_repl(ctx, input_fn=mock_input, out_fn=mock_out)

    assert any("streamed reply" in line for line in out_lines)


@pytest.mark.asyncio
async def test_run_repl_remove_signal_handler_exception() -> None:
    """Verifies run_repl handles exception when loop.remove_signal_handler fails."""
    ctx = create_harness(mock_provider=True)
    await ctx.flush()

    async def mock_stream_turn(session, prompt, **kwargs):
        yield {"type": "chunk", "text": "ok"}

    ctx.agent_loop.stream_turn = mock_stream_turn

    inputs = ["test", "/exit"]
    input_idx = 0

    def mock_input() -> str:
        nonlocal input_idx
        res = inputs[input_idx]
        input_idx += 1
        return res

    loop = asyncio.get_running_loop()
    with patch.object(loop, "remove_signal_handler", side_effect=RuntimeError("fail")):
        await run_repl(ctx, input_fn=mock_input, out_fn=lambda *a, **k: None)


@pytest.mark.asyncio
async def test_run_repl_sigint_handler_invocation() -> None:
    """Verifies that the registered SIGINT handler invokes sig.abort('SIGINT')."""
    ctx = create_harness(mock_provider=True)
    await ctx.flush()

    handler_ref: list[Any] = []

    def mock_add_signal_handler(sig_num: Any, callback: Any) -> None:
        handler_ref.append(callback)

    async def mock_stream_turn(session: Any, prompt: Any, **kwargs: Any) -> Any:
        if handler_ref:
            handler_ref[0]()
        sig = kwargs.get("signal")
        if sig:
            sig.throw_if_aborted()
        yield {"type": "chunk", "text": "never reached"}

    ctx.agent_loop.stream_turn = mock_stream_turn  # type: ignore[assignment]

    inputs = ["cancel turn", "/exit"]
    input_idx = 0

    def mock_input() -> str:
        nonlocal input_idx
        res = inputs[input_idx]
        input_idx += 1
        return res

    out_lines: list[str] = []
    loop = asyncio.get_running_loop()
    with patch.object(loop, "add_signal_handler", side_effect=mock_add_signal_handler):
        await run_repl(
            ctx,
            input_fn=mock_input,
            out_fn=lambda *a, **k: out_lines.append(" ".join(str(x) for x in a)),
        )

    assert any("Turn cancelled by user" in line for line in out_lines)


@pytest.mark.asyncio
async def test_run_repl_abort_error_and_turn_exception() -> None:
    """Verifies run_repl handling AbortError and generic Exception."""
    ctx = create_harness(mock_provider=True)
    await ctx.flush()

    inputs = ["abort me", "crash me", "/exit"]
    input_idx = 0

    def mock_input() -> str:
        nonlocal input_idx
        res = inputs[input_idx]
        input_idx += 1
        return res

    out_lines = []

    def mock_out(*args, **kwargs) -> None:
        out_lines.append(" ".join(str(a) for a in args))

    # Mock stream_turn to throw AbortError on first call, Exception on second call
    call_count = 0

    async def mock_stream_turn(session, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise AbortError("Cancelled")
        raise RuntimeError("LLM exploded")
        yield  # make it a generator

    ctx.agent_loop.stream_turn = mock_stream_turn

    await run_repl(ctx, input_fn=mock_input, out_fn=mock_out)

    assert any("Turn cancelled by user" in line for line in out_lines)
    assert any("Error during turn: LLM exploded" in line for line in out_lines)


@pytest.mark.asyncio
async def test_main_async_run_command(tmp_path: Path, capsys) -> None:
    """Verifies main_async with 'run' subcommand."""
    argv = [
        "--storage",
        str(tmp_path),
        "run",
        "Calculate 2+2",
    ]
    await main_async(argv)
    captured = capsys.readouterr().out
    assert "Running prompt: 'Calculate 2+2'" in captured
    assert "Assistant: Default mock response" in captured

    # Test with empty storage
    await main_async(["--storage", "", "run", "no storage run"])
    captured_empty = capsys.readouterr().out
    assert "Running prompt: 'no storage run'" in captured_empty


@pytest.mark.asyncio
async def test_main_async_session_commands(tmp_path: Path, capsys) -> None:
    """Verifies main_async with 'session list' and 'session show' subcommands."""
    # Create a persisted session first
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    s = ctx.sessions.create("persisted_123")
    ctx.sessions.log_event(s, EventType.USER_MESSAGE, {"text": "hello persisted"})

    # Test session list
    await main_async(["--storage", str(tmp_path), "session", "list"])
    captured_list = capsys.readouterr().out
    assert "persisted_123" in captured_list

    # Test session show (existing)
    await main_async(["--storage", str(tmp_path), "session", "show", "persisted_123"])
    captured_show = capsys.readouterr().out
    assert "hello persisted" in captured_show

    # Test session show (nonexistent)
    await main_async(["--storage", str(tmp_path), "session", "show", "missing_sess"])
    captured_missing = capsys.readouterr().out
    assert "Session not found: missing_sess" in captured_missing


@pytest.mark.asyncio
async def test_main_async_component_command(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """Verifies main_async with 'component list' subcommand."""
    from magitech_harness.components.manifest import ComponentManifest

    # Test when components are mounted
    old_create = create_harness

    def mock_create(*args, **kwargs):
        c = old_create(*args, **kwargs)
        manifest = ComponentManifest(
            name="test-listed-component",
            version="1.2.3",
            entry_point="dummy:apply",
        )

        class ExtraComponent:
            inject: ClassVar[list[str]] = ["components"]

            def __init__(self, ctx: Context) -> None:
                ctx.components._mounted_components["test-listed-component"] = (
                    manifest,
                    None,
                )

        c.use(ExtraComponent)
        return c

    monkeypatch.setattr("magitech_harness.cli.create_harness", mock_create)

    await main_async(["component", "list"])
    captured = capsys.readouterr().out
    assert "Loaded components" in captured
    assert "test-listed-component (1.2.3)" in captured


@pytest.mark.asyncio
async def test_main_async_profile_and_default_repl(tmp_path: Path, monkeypatch) -> None:
    """Verifies main_async with --profile flag and default REPL invocation."""
    import types

    dummy_pkg = types.ModuleType("dummy_provider_pkg")

    class DummyProviderComponent:
        inject: ClassVar[list[str]] = ["llm"]

        def __init__(self, c):
            c.llm.register_provider("dummy_p", MockLLMProvider())

    dummy_pkg.DummyProviderComponent = DummyProviderComponent
    monkeypatch.setitem(sys.modules, "dummy_provider_pkg", dummy_pkg)

    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text(
        "['dummy_provider_pkg:DummyProviderComponent']", encoding="utf-8"
    )

    repl_started = []

    async def mock_repl(ctx, model=None):
        repl_started.append(True)

    monkeypatch.setattr("magitech_harness.cli.run_repl", mock_repl)

    await main_async(["--profile", str(profile_file)])
    assert repl_started == [True]


def test_main_execution(monkeypatch) -> None:
    """Verifies main() wraps asyncio.run, and KeyboardInterrupt handling."""
    called = []

    def mock_run(coro):
        coro.close()
        called.append(True)

    monkeypatch.setattr("asyncio.run", mock_run)
    monkeypatch.setattr(sys, "argv", ["mth", "run", "hi"])
    main()
    assert called == [True]

    def mock_run_interrupt(coro):
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("asyncio.run", mock_run_interrupt)
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0


def test_main_module_execution(monkeypatch) -> None:
    """Verifies python -m magitech_harness.cli execution."""
    import runpy
    import warnings

    def mock_run(coro):
        coro.close()

    monkeypatch.setattr("asyncio.run", mock_run)
    monkeypatch.setattr(sys, "argv", ["mth"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("magitech_harness.cli", run_name="__main__")
