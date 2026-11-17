"""Unit tests for magitech_harness CLI entrypoint and interactive REPL."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from anima import Context, EventMode

import magitech_harness
from magitech_harness import check_bundle_entries, create_harness
from magitech_harness.agent.cancellation import AbortError
from magitech_harness.cli import (
    ReplDispatcher,
    main,
    main_async,
    run_repl,
)
from magitech_harness.llm.service import MockLLMProvider
from magitech_harness.session.events import EventType, SessionEvent


class _FakeReplAgent:
    """Minimal unified-driver agent that replays scripted REPL turn behavior."""

    def __init__(self, ctx: Any, session: Any, behavior: Any = None) -> None:
        self.session = session
        self._ctx = ctx
        self._behavior = behavior
        self.model = SimpleNamespace(current=lambda: None, select=lambda _sel: None)

    def followup(self, message: Any) -> None:
        if self._behavior is not None:
            self._behavior(self._ctx, self.session)

    async def when_idle(self) -> None:
        return None

    def cancel(self, cause: Any, options: Any = None) -> None:
        if self._behavior is not None and getattr(self._behavior, "on_cancel", None):
            self._behavior.on_cancel()


def _install_fake_repl_agent(ctx: Any, behavior: Any = None) -> Any:
    """Patch ``ctx.agents.create`` to return a scripted unified-driver agent."""
    created: list[Any] = []

    async def _fake_create(options: Any) -> Any:
        session = ctx.sessions.create(str(options.get("sessionId", "repl-fake")))
        agent = _FakeReplAgent(ctx, session, behavior)
        created.append(agent)
        return {"agent": agent, "dispose": lambda: None}

    ctx.agents.create = _fake_create  # type: ignore[method-assign]
    return created


def _emit_stream(ctx: Any, text: str) -> None:
    ctx.emit(
        "agent/assistant-stream",
        {
            "frame": {
                "type": "chunk",
                "chunk": {"type": "text-delta", "index": 0, "text": text},
            }
        },
    )


class _MockProviderComponent:
    """Test-only component registering the offline mock provider."""

    inject: ClassVar[list[str]] = ["llm"]

    def __init__(self, ctx: Context) -> None:
        ctx.llm.register_provider("mock", MockLLMProvider())


@pytest.mark.asyncio
async def test_repl_dispatcher_all_commands(tmp_path: Path) -> None:
    """Verifies ReplDispatcher handling of all slash commands."""
    ctx = create_harness(storage_dir=tmp_path)
    ctx.use(_MockProviderComponent)
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
    """Verifies run_repl renders streamed chunks and tool events from the bus."""
    ctx = create_harness()
    await ctx.flush()

    def _behavior(repl_ctx: Any, repl_session: Any) -> None:
        _emit_stream(repl_ctx, "Final answer")
        repl_ctx.emit(
            "session/event",
            repl_session,
            SessionEvent(
                id="c1",
                session_id=repl_session.id,
                type=EventType.TOOL_CALL,
                payload={"name": "dummy_tool", "arguments": {"val": 1}},
            ),
        )
        repl_ctx.emit(
            "session/event",
            repl_session,
            SessionEvent(
                id="r1",
                session_id=repl_session.id,
                type=EventType.TOOL_RESULT,
                payload={"name": "dummy_tool", "is_error": False},
            ),
        )

    _install_fake_repl_agent(ctx, behavior=_behavior)

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

    out_lines: list[str] = []

    def mock_out(*args: Any, **kwargs: Any) -> None:
        out_lines.append(" ".join(str(a) for a in args))

    await run_repl(ctx, input_fn=mock_input, out_fn=mock_out)

    assert any("Tool Start" in line for line in out_lines)
    assert any("OK" in line for line in out_lines)
    assert any("Final answer" in line for line in out_lines)


@pytest.mark.asyncio
async def test_run_repl_signal_handler_unsupported_and_usage_event() -> None:
    """Verifies run_repl when loop.add_signal_handler raises NotImplementedError."""
    ctx = create_harness()
    await ctx.flush()
    _install_fake_repl_agent(
        ctx, behavior=lambda repl_ctx, _s: _emit_stream(repl_ctx, "streamed reply")
    )

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
    ctx = create_harness()
    await ctx.flush()
    _install_fake_repl_agent(
        ctx, behavior=lambda repl_ctx, _s: _emit_stream(repl_ctx, "ok")
    )

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
    """Verifies that the registered SIGINT handler cancels the agent turn."""
    ctx = create_harness()
    await ctx.flush()

    handler_ref: list[Any] = []
    cancelled = {"value": False}

    def mock_add_signal_handler(sig_num: Any, callback: Any) -> None:
        handler_ref.append(callback)

    def _behavior(repl_ctx: Any, _s: Any) -> None:
        if handler_ref:
            handler_ref[0]()

    _install_fake_repl_agent(ctx, behavior=_behavior)

    original_create = ctx.agents.create

    async def _wrapped_create(options: Any) -> Any:
        handle = await original_create(options)
        agent = handle["agent"]

        def _on_cancel(cause: Any = None) -> None:
            cancelled["value"] = True

        async def _abort_when_idle() -> None:
            if cancelled["value"]:
                raise AbortError("Cancelled by SIGINT")

        agent.cancel = _on_cancel  # type: ignore[method-assign]
        agent.when_idle = _abort_when_idle  # type: ignore[method-assign]
        return handle

    ctx.agents.create = _wrapped_create  # type: ignore[method-assign]

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
async def test_run_repl_ignores_foreign_session_and_non_text_chunks() -> None:
    """REPL ignores other sessions' events and non-text stream frames."""
    ctx = create_harness()
    await ctx.flush()
    other = ctx.sessions.create("other-session")

    def _behavior(repl_ctx: Any, repl_session: Any) -> None:
        # foreign session event (should be ignored)
        repl_ctx.emit(
            "session/event",
            other,
            SessionEvent(
                id="x",
                session_id=other.id,
                type=EventType.TOOL_CALL,
                payload={"name": "foreign"},
            ),
        )
        # non-chunk frame (should be ignored)
        repl_ctx.emit("agent/assistant-stream", {"frame": {"type": "start"}})
        # chunk frame with a non-dict chunk (should be ignored)
        repl_ctx.emit(
            "agent/assistant-stream",
            {"frame": {"type": "chunk", "chunk": "not-a-dict"}},
        )
        _emit_stream(repl_ctx, "shown")

    _install_fake_repl_agent(ctx, behavior=_behavior)
    out_lines: list[str] = []

    inputs = ["go", "/exit"]
    idx = [0]

    def mock_input() -> str:
        r = inputs[idx[0]]
        idx[0] += 1
        return r

    await run_repl(
        ctx,
        input_fn=mock_input,
        out_fn=lambda *a, **k: out_lines.append(" ".join(str(x) for x in a)),
    )
    assert any("shown" in line for line in out_lines)
    assert not any("foreign" in line for line in out_lines)


@pytest.mark.asyncio
async def test_run_repl_abort_error_and_turn_exception() -> None:
    """Verifies run_repl handling AbortError and generic Exception from when_idle."""
    ctx = create_harness()
    await ctx.flush()

    inputs = ["abort me", "crash me", "/exit"]
    input_idx = 0

    def mock_input() -> str:
        nonlocal input_idx
        res = inputs[input_idx]
        input_idx += 1
        return res

    out_lines: list[str] = []

    def mock_out(*args: Any, **kwargs: Any) -> None:
        out_lines.append(" ".join(str(a) for a in args))

    call_count = 0

    async def _flaky_idle() -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise AbortError("Cancelled")
        raise RuntimeError("LLM exploded")

    original_create = ctx.agents.create

    async def _wrapped_create(options: Any) -> Any:
        handle = await original_create(options)
        handle["agent"].when_idle = _flaky_idle  # type: ignore[method-assign]
        return handle

    ctx.agents.create = _wrapped_create  # type: ignore[method-assign]

    # Force SIGINT handler installation to succeed so the second turn
    # deterministically exercises the already-installed path on all OSes.
    loop = asyncio.get_running_loop()
    with (
        patch.object(loop, "add_signal_handler", return_value=None),
        patch.object(loop, "remove_signal_handler", return_value=None),
    ):
        await run_repl(ctx, input_fn=mock_input, out_fn=mock_out)

    assert any("Turn cancelled by user" in line for line in out_lines)
    assert any("Error during turn: LLM exploded" in line for line in out_lines)


@pytest.mark.asyncio
async def test_main_async_fails_loud_on_bad_bundle(tmp_path: Path, monkeypatch) -> None:
    """Verifies the CLI boot audit fails loud on a bad bundle row."""
    bundle_path = tmp_path / "bundle.yaml"
    bundle_path.write_text(
        "- id: bad-row\n  name: nonexistent.module:Nope\n  config: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    with pytest.raises(RuntimeError, match="bad-row"):
        await main_async(["--storage", str(tmp_path), "session", "list"])


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
    await ctx.sessions.flush()

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
    # Bundle service rows are listed too, including Phase 3/4 services.
    assert "Bundle service rows" in captured
    assert "- context [bundle row]" in captured
    assert "- compaction [bundle row]" in captured

    # A context without a loader skips the bundle-row section.
    def mock_create_no_loader(*args, **kwargs):
        c = old_create(*args, **kwargs)
        real_get = c.get

        def get_without_loader(name, *a, **k):
            if name == "loader":
                return None
            return real_get(name, *a, **k)

        c.get = get_without_loader  # type: ignore[method-assign]
        return c

    monkeypatch.setattr("magitech_harness.cli.create_harness", mock_create_no_loader)

    # The boot audit requires a loader; skip it for this loader-less probe.
    async def noop_check(_ctx: Context) -> None:
        return None

    monkeypatch.setattr("magitech_harness.cli.check_bundle_entries", noop_check)
    await main_async(["component", "list"])
    captured = capsys.readouterr().out
    assert "Bundle service rows" not in captured


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


@pytest.mark.asyncio
async def test_main_async_boot_order_bundle_profile_fallback_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    """Verifies the CLI boots bundle rows, then the user profile, then dispatch.

    The user profile registers a provider, so the mock fallback must be
    skipped: dispatch sees the profile provider and no mock provider.
    """
    import types

    order: list[str] = []
    seen: dict[str, Any] = {}

    real_create = create_harness

    def spy_create(*args, **kwargs):
        order.append("create_harness")
        seen["profile"] = kwargs.get("profile")
        return real_create(*args, **kwargs)

    monkeypatch.setattr("magitech_harness.cli.create_harness", spy_create)

    def spy_audit(ctx):
        order.append("check_bundle_entries")
        return check_bundle_entries(ctx)

    monkeypatch.setattr("magitech_harness.cli.check_bundle_entries", spy_audit)

    dummy_pkg = types.ModuleType("k5_order_provider_pkg")

    class DummyProviderComponent:
        inject: ClassVar[list[str]] = ["llm"]

        def __init__(self, c):
            c.llm.register_provider("dummy_p", MockLLMProvider())

    dummy_pkg.DummyProviderComponent = DummyProviderComponent
    monkeypatch.setitem(sys.modules, "k5_order_provider_pkg", dummy_pkg)

    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text(
        "['k5_order_provider_pkg:DummyProviderComponent']", encoding="utf-8"
    )

    async def spy_repl(ctx, model=None):
        order.append("dispatch")
        seen["providers"] = set(ctx.llm.providers)

    monkeypatch.setattr("magitech_harness.cli.run_repl", spy_repl)

    await main_async(["--storage", str(tmp_path), "--profile", str(profile_file)])
    assert order == [
        "create_harness",
        "check_bundle_entries",
        "dispatch",
    ]
    assert seen["profile"] == str(profile_file)
    assert seen["providers"] == {"dummy_p"}


@pytest.mark.asyncio
async def test_main_async_mock_fallback_runs_before_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    """Verifies the no-provider mock fallback registers before dispatch."""
    seen: dict[str, Any] = {}

    async def spy_repl(ctx, model=None):
        seen["providers"] = set(ctx.llm.providers)

    monkeypatch.setattr("magitech_harness.cli.run_repl", spy_repl)

    await main_async(["--storage", str(tmp_path)])
    assert seen["providers"] == {"mock"}


@pytest.mark.asyncio
async def test_main_async_drains_titles_before_flush(
    tmp_path: Path, monkeypatch
) -> None:
    """Verifies CLI drains SessionTitleService before flushing sessions on exit."""
    order: list[str] = []
    real_create = create_harness

    def spy_create(*args, **kwargs):
        ctx = real_create(*args, **kwargs)
        orig_ctx_flush = ctx.flush

        async def wrapped_ctx_flush() -> None:
            await orig_ctx_flush()
            if (
                "title_hooked" not in order
                and ctx.get("sessionTitle", None, strict=False) is not None
            ):
                order.append("title_hooked")
                orig_drain = ctx.sessionTitle.drain
                orig_flush = ctx.sessions.flush

                async def traced_drain() -> None:
                    order.append("title_drain")
                    await orig_drain()

                async def traced_flush(*f_args, **f_kwargs) -> None:
                    order.append("sessions_flush")
                    await orig_flush(*f_args, **f_kwargs)

                ctx.sessionTitle.drain = traced_drain  # type: ignore[method-assign]
                ctx.sessions.flush = traced_flush  # type: ignore[method-assign]

        ctx.flush = wrapped_ctx_flush  # type: ignore[method-assign]
        return ctx

    monkeypatch.setattr("magitech_harness.cli.create_harness", spy_create)
    await main_async(["--storage", str(tmp_path), "run", "hello world"])
    assert "title_drain" in order
    assert "sessions_flush" in order
    assert order.index("title_drain") < order.index("sessions_flush")


def test_summarize_headless_outcome_and_exit_codes() -> None:
    """Verifies headless interval summary and exit code mapping."""
    from magitech_harness.cli import headless_exit_code, summarize_headless_outcome
    from magitech_harness.llm.message import create_user_message
    from magitech_harness.session.store import SessionStore

    store = SessionStore()
    # Events before turn/start are ignored.
    session = store.create("h1")
    session.append(
        EventType.USER_MESSAGE,
        create_user_message(
            content=[{"type": "text", "text": "pre"}], source={"kind": "user"}
        ),
        surface_op="append",
    )
    # Empty interval -> no turn -> exit code 1.
    assert headless_exit_code(summarize_headless_outcome(session, 0)) == 1

    session.append(EventType.TURN_START, {"turn": 1})
    # Non-assistant event inside a turn (e.g. step/start) is skipped cleanly.
    session.append(EventType.STEP_START, {"turn": 1, "step": 1})
    # Empty text block / missing content does not overwrite text.
    session.append(
        EventType.ASSISTANT_MESSAGE,
        {
            "turn": 1,
            "step": 1,
            "stream": [],
            "message": {
                "id": "msg-0",
                "role": "assistant",
                "content": [],
                "source": {"kind": "model", "provider": "p", "model": "m"},
            },
        },
        surface_op="append",
    )
    session.append(
        EventType.ASSISTANT_MESSAGE,
        {
            "turn": 1,
            "step": 1,
            "stream": [],
            "message": {
                "id": "msg-1",
                "role": "assistant",
                "content": [{"type": "text", "text": "one"}],
                "source": {"kind": "model", "provider": "p", "model": "m"},
            },
        },
        surface_op="append",
    )
    session.append(EventType.TURN_END, {"turn": 1, "reason": {"kind": "completed"}})

    # Non-list / malformed message content is ignored defensively.
    from types import SimpleNamespace

    synthetic = SimpleNamespace(
        events=[
            session.events[1],
            type(session.events[2])(
                id="fake-1",
                session_id="h1",
                type=EventType.ASSISTANT_MESSAGE,
                timestamp=0.0,
                payload={"turn": 1, "step": 1, "message": "not-a-dict"},
            ),
            session.events[-1],
        ]
    )
    assert summarize_headless_outcome(synthetic, 0).text == ""  # type: ignore[arg-type]
    first_done = summarize_headless_outcome(session, 0)
    assert first_done.text == "one"
    assert headless_exit_code(first_done) == 0

    # Unclosed second turn in the interval resets reason so completed is NOT inherited!
    session.append(EventType.TURN_START, {"turn": 2})
    unclosed = summarize_headless_outcome(session, 0)
    assert unclosed.reason is None
    assert headless_exit_code(unclosed) == 1

    # Non-completed end reasons exit 1.
    for kind in ("aborted", "blocked", "error", "max-tokens", "interrupted", "forked"):
        s2 = store.create(f"h-{kind}")
        s2.append(EventType.TURN_START, {"turn": 1})
        s2.append(EventType.TURN_END, {"turn": 1, "reason": {"kind": kind}})
        assert headless_exit_code(summarize_headless_outcome(s2, 0)) == 1


@pytest.mark.asyncio
async def test_cli_headless_subcommand_exit_codes_and_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Verifies `mth headless` drives AgentLoop and returns 0/1 by turn outcome."""
    import types
    from collections.abc import AsyncIterator
    from typing import cast

    from magitech_harness.llm.service import LlmAdapter
    from magitech_harness.llm.types import (
        AbortSignal,
        GenerateOptions,
        LlmModelInfo,
        LlmResolvedModelInfo,
        StreamChunk,
    )

    class _OkAdapter(LlmAdapter):
        seen_models: ClassVar[list[str]] = []

        async def list_models(self, provider: str) -> list[LlmModelInfo]:
            del provider
            return [
                {
                    "provider": "ok",
                    "id": "m1",
                    "name": "m1",
                    "reasoning": {
                        "defaultEffort": "medium",
                        "efforts": [
                            {"id": "medium", "name": "Medium"},
                            {"id": "high", "name": "High"},
                        ],
                    },
                },
                {"provider": "ok", "id": "m2", "name": "m2"},
            ]

        async def resolve_model(
            self,
            provider: str,
            model: str,
            signal: AbortSignal | None = None,
        ) -> LlmResolvedModelInfo:
            del signal
            for info in await self.list_models(provider):
                if info["id"] == model:
                    return cast(LlmResolvedModelInfo, dict(info))
            return {"provider": provider, "id": model, "name": model}

        async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
            _OkAdapter.seen_models.append(str(options.get("model", "")))
            yield {"type": "block-start", "index": 0, "blockType": "text"}
            yield {"type": "text-delta", "index": 0, "text": "Headless answer"}
            yield {
                "type": "block-end",
                "index": 0,
                "block": {"type": "text", "text": "Headless answer"},
            }
            yield {"type": "finish", "reason": {"kind": "stop"}}

    class _ErrAdapter(LlmAdapter):
        async def list_models(self, provider: str) -> list[LlmModelInfo]:
            del provider
            return [{"provider": "err", "id": "m1", "name": "m1"}]

        async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
            del options
            raise RuntimeError("boom")
            yield {"type": "finish", "finishReason": "error"}  # pragma: no cover

    mod = types.ModuleType("headless_test_adapters")

    class _OkPlugin:
        inject: ClassVar[list[str]] = ["llm"]

        def __init__(self, c: Any) -> None:
            c.llm.register_adapter(["ok"], _OkAdapter())

    class _ErrPlugin:
        inject: ClassVar[list[str]] = ["llm"]

        def __init__(self, c: Any) -> None:
            c.llm.register_adapter(["err"], _ErrAdapter())

    mod.OkPlugin = _OkPlugin
    mod.ErrPlugin = _ErrPlugin
    monkeypatch.setitem(sys.modules, "headless_test_adapters", mod)

    ok_profile = tmp_path / "ok.yaml"
    ok_profile.write_text("['headless_test_adapters:OkPlugin']", encoding="utf-8")
    err_profile = tmp_path / "err.yaml"
    err_profile.write_text("['headless_test_adapters:ErrPlugin']", encoding="utf-8")

    ok_store = tmp_path / "ok_store"
    err_store = tmp_path / "err_store"

    code_ok = await main_async(
        [
            "--storage",
            str(ok_store),
            "--profile",
            str(ok_profile),
            "--model",
            "ok/m1",
            "headless",
            "hi",
        ]
    )
    assert code_ok == 0
    captured = capsys.readouterr()
    assert "Headless answer" in captured.out

    code_err = await main_async(
        [
            "--storage",
            str(err_store),
            "--profile",
            str(err_profile),
            "headless",
            "hi",
        ]
    )
    assert code_err == 1
    assert "mth:" in capsys.readouterr().err

    # Also test --json output + --session-id resume + model override on resume + reasoning preserve.
    import json as py_json

    resumed_sess = next(ok_store.rglob("session.v4.jsonl")).parent.name
    # Seed a model/selection with reasoningEffort='high' on ok/m1 so same-model resume preserves it.
    from magitech_harness.cli import _run_headless

    ctx_seed = create_harness(storage_dir=ok_store)
    await ctx_seed.flush()
    h_seed = await ctx_seed.agents.resume({"resumeSessionId": resumed_sess})
    try:
        h_seed["agent"].model.select(
            {"provider": "ok", "model": "m1", "reasoningEffort": "high"}
        )
        await ctx_seed.sessions.flush(h_seed["agent"].session)
    finally:
        await h_seed["dispose"]()

    _OkAdapter.seen_models.clear()
    # Resume with same ok/m1 preserves reasoningEffort; resume with ok/m2 overrides model.
    code_same = await main_async(
        [
            "--storage",
            str(ok_store),
            "--profile",
            str(ok_profile),
            "--model",
            "ok/m1",
            "headless",
            "--session-id",
            resumed_sess,
            "same-model",
        ]
    )
    assert code_same == 0
    capsys.readouterr()

    code_json = await main_async(
        [
            "--storage",
            str(ok_store),
            "--profile",
            str(ok_profile),
            "--model",
            "ok/m2",
            "headless",
            "--session-id",
            resumed_sess,
            "--json",
            "again",
        ]
    )
    assert code_json == 0
    assert _OkAdapter.seen_models == ["m1", "m2"]
    json_out = capsys.readouterr().out.strip().splitlines()[-1]
    payload = py_json.loads(json_out)
    assert payload["exitCode"] == 0
    assert payload["text"] == "Headless answer"

    # Resume with --model omitted keeps existing selection without clobbering.
    code_keep = await main_async(
        [
            "--storage",
            str(ok_store),
            "--profile",
            str(ok_profile),
            "headless",
            "--session-id",
            resumed_sess,
            "keep",
        ]
    )
    assert code_keep == 0
    capsys.readouterr()

    # Blank session_id and blank task fail closed with stderr + --json error object.
    ctx_fail = create_harness(storage_dir=tmp_path / "fail_store")
    await ctx_fail.flush()
    assert await _run_headless(ctx_fail, "hi", session_id="   ", json_mode=True) == 1
    err_cap = capsys.readouterr()
    assert py_json.loads(err_cap.out.strip())["type"] == "error"
    assert "session_id must not be blank" in err_cap.err

    assert await _run_headless(ctx_fail, "   ", json_mode=False) == 1
    assert "a task is required" in capsys.readouterr().err

    # Uncleared agent/error without turn/end, plus synchronous dispose() and runtime exception in body.
    from types import SimpleNamespace

    class _FakeAgent:
        def __init__(self, s: Any, trigger: Any) -> None:
            self.session = s
            self._trigger = trigger
            self.model = SimpleNamespace(current=lambda: None, select=lambda _sel: None)

        def followup(self, _msg: Any) -> None:
            self._trigger()

        async def when_idle(self) -> None:
            return None

    fake_ctx = create_harness(storage_dir=tmp_path / "fake_store")
    await fake_ctx.flush()
    fake_sess = fake_ctx.sessions.create("fake-err")
    other_sess = fake_ctx.sessions.create("other-sess")
    dispose_calls: list[str] = []

    def _emit_agent_err() -> None:
        fake_ctx.emit(
            "agent/error",
            {"agent": SimpleNamespace(), "error": RuntimeError("ignored")},
        )
        fake_ctx.emit(
            "agent/error", {"agent": fake_agent, "error": RuntimeError("orphan crash")}
        )
        # Non-turn/end event and wrong-session turn/end must not clear failure.
        fake_ctx.emit(
            "session/event",
            fake_sess,
            SimpleNamespace(type=EventType.STEP_START, seq=99),
        )
        fake_ctx.emit(
            "session/event",
            other_sess,
            SimpleNamespace(type=EventType.TURN_END, seq=99),
        )

    fake_agent = _FakeAgent(fake_sess, _emit_agent_err)

    async def _fake_create(_opts: Any) -> Any:
        return {"agent": fake_agent, "dispose": lambda: dispose_calls.append("sync")}

    fake_ctx.agents.create = _fake_create  # type: ignore[method-assign]
    assert await _run_headless(fake_ctx, "hi", json_mode=True) == 1
    assert dispose_calls == ["sync"]
    orphan_out = capsys.readouterr()
    assert "orphan crash" in orphan_out.out
    assert "orphan crash" in orphan_out.err

    # Exception raised after handle creation inside followup/when_idle hits inner except + async/sync dispose.
    def _boom_trigger() -> None:
        raise RuntimeError("followup boom")

    fake_agent2 = _FakeAgent(fake_sess, _boom_trigger)

    async def _fake_create2(_opts: Any) -> Any:
        return {"agent": fake_agent2, "dispose": lambda: dispose_calls.append("sync2")}

    fake_ctx.agents.create = _fake_create2  # type: ignore[method-assign]
    assert await _run_headless(fake_ctx, "hi") == 1
    assert dispose_calls == ["sync", "sync2"]
    assert "followup boom" in capsys.readouterr().err

    # Also test turn/end error whose error payload is non-dict (defensive branch).
    from magitech_harness.session.events import SessionEvent

    fake_sess_non_dict = fake_ctx.sessions.create("non-dict-err")

    def _emit_non_dict_turn() -> None:
        fake_sess_non_dict.events.extend(
            [
                SessionEvent(
                    id="t1",
                    session_id="non-dict-err",
                    type=EventType.TURN_START,
                    timestamp=0.0,
                    payload={"turn": 1},
                ),
                SessionEvent(
                    id="t2",
                    session_id="non-dict-err",
                    type=EventType.TURN_END,
                    timestamp=0.0,
                    payload={
                        "turn": 1,
                        "reason": {"kind": "error", "error": "plain-str"},
                    },
                ),
            ]
        )

    fake_agent3 = _FakeAgent(fake_sess_non_dict, _emit_non_dict_turn)

    async def _fake_create3(_opts: Any) -> Any:
        return {"agent": fake_agent3, "dispose": lambda: None}

    fake_ctx.agents.create = _fake_create3  # type: ignore[method-assign]
    assert await _run_headless(fake_ctx, "hi") == 1

    # Cover default provider "mock" fallback when neither adapter nor --model/--profile exist,
    # and _drain_and_flush when sessionTitle is absent.
    from magitech_harness.cli import _drain_and_flush, _resolve_headless_route
    from magitech_harness.llm.service import LLMService

    bare = Context()
    LLMService(bare)
    assert _resolve_headless_route(bare, None) == ("mock", "mock")

    # Test main() exits 1 when main_async returns non-zero.
    def mock_run_err(coro):
        coro.close()
        return 1

    monkeypatch.setattr("asyncio.run", mock_run_err)
    monkeypatch.setattr(sys, "argv", ["mth", "headless", "hi"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

    # Test _drain_and_flush without sessionTitle mounted.
    class _SessStub:
        async def flush(self) -> None:
            return None

    class _MinCtx:
        sessions = _SessStub()

        def get(self, name: str, default: Any = None, strict: bool = True) -> Any:
            del name, strict
            return default

    await _drain_and_flush(_MinCtx())  # type: ignore[arg-type]
