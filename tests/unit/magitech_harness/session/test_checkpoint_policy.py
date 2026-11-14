"""Unit tests for the durability checkpoint policy.

The policy flushes the session log before model dispatch, before
top-level tool dispatch, and before each step, and never flushes at
``turn/end``. A flush rejection blocks dispatch (fail-closed).
"""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from magitech_harness import create_harness
from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.types import ModelChunk, ModelResponse
from magitech_harness.session.events import EventType, ToolCallData
from magitech_harness.tools.definition import define_tool

CANONICAL_ABORT_TEXT = "Error: tool call aborted before dispatch"
CANONICAL_ABORT_ERROR = {
    "message": "tool call aborted before dispatch",
    "info": {"name": "AbortError", "code": "ABORTED_BEFORE_DISPATCH"},
}


class RecordingProvider:
    """Provider that records adapter invocations in call order."""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls = 0

    async def generate(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ModelResponse:
        return ModelResponse(text="hello")

    async def stream(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[ModelChunk]:
        self.calls += 1
        self.order.append("adapter")
        yield ModelChunk(text="hello")


def _install_recording_provider(ctx: Any, order: list[str]) -> RecordingProvider:
    """Register the recording provider as the active provider."""
    provider = RecordingProvider(order)
    ctx.llm.register_provider("rec", provider)
    ctx.llm.set_active_provider("rec")
    return provider


async def _drain(stream: Any) -> list[Any]:
    """Collect every chunk from a stream."""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
async def test_llm_stream_flushes_before_first_chunk(tmp_path: Path) -> None:
    """Verifies the flush completes before the adapter runs or a chunk yields."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    provider = _install_recording_provider(ctx, order)
    session = ctx.sessions.create("llm-checkpoint")
    started = asyncio.Event()
    gate = asyncio.Event()

    async def gated_flush(flushed: Any) -> None:
        order.append("flush:start")
        started.set()
        await gate.wait()
        order.append("flush:end")

    ctx.on("session/flush", gated_flush)

    pending = asyncio.create_task(
        _drain(
            ctx.llm.stream(messages=[], session_id=session.session_id),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    # The flush is in flight: the adapter must not have run yet.
    assert order == ["flush:start"]
    assert provider.calls == 0
    gate.set()
    chunks = await asyncio.wait_for(pending, timeout=5)
    assert order == ["flush:start", "flush:end", "adapter"]
    assert [chunk.text for chunk in chunks] == ["hello"]


@pytest.mark.asyncio
async def test_llm_stream_without_session_id_passes_through(tmp_path: Path) -> None:
    """Verifies a request naming no session dispatches without checkpointing."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    provider = _install_recording_provider(ctx, order)
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))

    chunks = await _drain(ctx.llm.stream(messages=[]))

    assert flushes == []
    assert provider.calls == 1
    assert [chunk.text for chunk in chunks] == ["hello"]


@pytest.mark.asyncio
async def test_llm_stream_unknown_session_passes_through(tmp_path: Path) -> None:
    """Verifies a request naming an unknown session dispatches uncheckpointed."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    provider = _install_recording_provider(ctx, order)
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))

    chunks = await _drain(ctx.llm.stream(messages=[], session_id="detached"))

    assert flushes == []
    assert provider.calls == 1
    assert [chunk.text for chunk in chunks] == ["hello"]


@pytest.mark.asyncio
async def test_llm_stream_failing_flush_never_dispatches(tmp_path: Path) -> None:
    """Verifies a rejected checkpoint prevents adapter dispatch entirely."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    provider = _install_recording_provider(ctx, order)
    session = ctx.sessions.create("llm-failure")

    async def failing_flush(flushed: Any) -> None:
        raise RuntimeError("disk unavailable")

    ctx.on("session/flush", failing_flush)

    with pytest.raises(RuntimeError, match="disk unavailable"):
        await _drain(ctx.llm.stream(messages=[], session_id=session.session_id))
    assert order == []
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_llm_stream_log_is_durable_before_first_chunk(tmp_path: Path) -> None:
    """Verifies the logged prefix reaches disk before the first chunk yields."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    _install_recording_provider(ctx, order)
    session = ctx.sessions.create("llm-durable")
    ctx.sessions.log_event(session, EventType.TURN_START, {"turn_id": "turn-1"})

    stream = ctx.llm.stream(messages=[], session_id=session.session_id)
    first = await stream.__aiter__().__anext__()
    assert first.text == "hello"
    await stream.aclose()

    location = ctx.sessions.location(session.session_id)
    assert location is not None
    lines = location.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["type"] == "session"
    assert any("turn/start" in line for line in lines[1:])


@pytest.mark.asyncio
async def test_tool_execute_flushes_before_top_level_body(tmp_path: Path) -> None:
    """Verifies a top-level tool call flushes before its body runs."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    session = ctx.sessions.create("tool-checkpoint")
    started = asyncio.Event()
    gate = asyncio.Event()

    async def gated_flush(flushed: Any) -> None:
        order.append("flush:start")
        started.set()
        await gate.wait()
        order.append("flush:end")

    ctx.on("session/flush", gated_flush)
    ctx.tools.register(
        define_tool(
            name="write",
            description="side effect",
            execute=lambda args: order.append("tool") or "done",
        )
    )

    pending = asyncio.create_task(
        ctx.tools.execute(
            ToolCallData(call_id="write-1", name="write", arguments={}),
            session=session,
            signal=AbortSignal(),
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    assert order == ["flush:start"]
    gate.set()
    result = await asyncio.wait_for(pending, timeout=5)
    assert order == ["flush:start", "flush:end", "tool"]
    assert not result.is_error
    assert result.content == "done"


@pytest.mark.asyncio
async def test_tool_execute_failing_flush_never_runs_body(tmp_path: Path) -> None:
    """Verifies a rejected checkpoint yields an error result, body unrun."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("tool-failure")
    ran = False

    async def failing_flush(flushed: Any) -> None:
        raise RuntimeError("disk unavailable")

    ctx.on("session/flush", failing_flush)

    def body(args: dict[str, Any]) -> str:
        nonlocal ran
        ran = True
        return "done"

    ctx.tools.register(
        define_tool(name="write", description="side effect", execute=body)
    )

    result = await ctx.tools.execute(
        ToolCallData(call_id="write-2", name="write", arguments={}),
        session=session,
        signal=AbortSignal(),
    )

    assert result.is_error
    assert "disk unavailable" in result.content
    assert ran is False


@pytest.mark.asyncio
async def test_tool_execute_abort_during_flush_returns_canonical_result(
    tmp_path: Path,
) -> None:
    """Verifies an abort landing during the flush yields the canonical result."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    session = ctx.sessions.create("tool-checkpoint-cancel")
    signal = AbortSignal()
    started = asyncio.Event()
    gate = asyncio.Event()

    async def gated_flush(flushed: Any) -> None:
        order.append("flush:start")
        started.set()
        await gate.wait()
        order.append("flush:end")

    ctx.on("session/flush", gated_flush)

    def body(args: dict[str, Any]) -> str:
        order.append("tool")
        return "done"

    ctx.tools.register(
        define_tool(name="write", description="side effect", execute=body)
    )

    pending = asyncio.create_task(
        ctx.tools.execute(
            ToolCallData(call_id="write-cancelled", name="write", arguments={}),
            session=session,
            signal=signal,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=5)
    assert order == ["flush:start"]
    signal.abort("cancelled during checkpoint")
    gate.set()
    result = await asyncio.wait_for(pending, timeout=5)

    assert result.is_error
    assert result.content == CANONICAL_ABORT_TEXT
    assert result.error == CANONICAL_ABORT_ERROR
    assert order == ["flush:start", "flush:end"]


@pytest.mark.asyncio
async def test_tool_execute_aborted_signal_before_flush_returns_canonical_result(
    tmp_path: Path,
) -> None:
    """Verifies a pre-aborted signal still flushes first, then returns canonical."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("tool-pre-aborted")
    signal = AbortSignal()
    signal.abort("already cancelled")
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))
    ran = False

    def body(args: dict[str, Any]) -> str:
        nonlocal ran
        ran = True
        return "done"

    ctx.tools.register(
        define_tool(name="write", description="side effect", execute=body)
    )

    result = await ctx.tools.execute(
        ToolCallData(call_id="write-3", name="write", arguments={}),
        session=session,
        signal=signal,
    )

    assert flushes == [session.session_id]
    assert result.is_error
    assert result.content == CANONICAL_ABORT_TEXT
    assert result.error == CANONICAL_ABORT_ERROR
    assert ran is False


@pytest.mark.asyncio
async def test_tool_execute_without_session_passes_through(tmp_path: Path) -> None:
    """Verifies a tool call naming no session dispatches without checkpointing."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))
    ran = False

    def body(args: dict[str, Any]) -> str:
        nonlocal ran
        ran = True
        return "done"

    ctx.tools.register(
        define_tool(name="write", description="side effect", execute=body)
    )

    result = await ctx.tools.execute(
        ToolCallData(call_id="write-4", name="write", arguments={}),
    )

    assert flushes == []
    assert not result.is_error
    assert ran is True


@pytest.mark.asyncio
async def test_tool_execute_nested_call_reuses_outer_checkpoint(
    tmp_path: Path,
) -> None:
    """Verifies a nested dispatch (explicit parent) performs no extra flush."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("nested-tool")
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))
    ctx.tools.register(
        define_tool(name="nested", description="nested", execute=lambda args: "done")
    )

    result = await ctx.tools.execute(
        ToolCallData(call_id="nested-1", name="nested", arguments={}),
        session=session,
        signal=AbortSignal(),
        parent=object(),
    )

    assert flushes == []
    assert not result.is_error


@pytest.mark.asyncio
async def test_tool_execute_reentrant_body_reuses_outer_checkpoint(
    tmp_path: Path,
) -> None:
    """Verifies a body that re-enters execute flushes once, for the outer call."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("reentrant-tool")
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))
    signal = AbortSignal()

    async def outer_body(args: dict[str, Any]) -> str:
        inner = await ctx.tools.execute(
            ToolCallData(call_id="inner-1", name="inner", arguments={}),
            session=session,
            signal=signal,
        )
        assert not inner.is_error
        return "outer-done"

    ctx.tools.register(
        define_tool(name="outer", description="outer", execute=outer_body)
    )
    ctx.tools.register(
        define_tool(
            name="inner", description="inner", execute=lambda args: "inner-done"
        )
    )

    result = await ctx.tools.execute(
        ToolCallData(call_id="outer-1", name="outer", arguments={}),
        session=session,
        signal=signal,
    )

    assert flushes == [session.session_id]
    assert not result.is_error
    assert result.content == "outer-done"


@pytest.mark.asyncio
async def test_pre_step_flushes_before_step(tmp_path: Path) -> None:
    """Verifies the pre-step waterfall flushes the step's session first."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("pre-step")
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))

    decision = await ctx.waterfall(
        "agent/pre-step",
        {"action": "continue", "step_index": 1, "session": session},
        default_handler=lambda decision: decision,
    )

    assert decision["action"] == "continue"
    assert flushes == [session.session_id]


@pytest.mark.asyncio
async def test_pre_step_without_session_passes_through(tmp_path: Path) -> None:
    """Verifies a pre-step decision naming no session passes through unflushed."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))

    decision = await ctx.waterfall(
        "agent/pre-step",
        {"action": "continue", "step_index": 1},
        default_handler=lambda decision: decision,
    )

    assert decision["action"] == "continue"
    assert flushes == []


@pytest.mark.asyncio
async def test_pre_step_failing_flush_never_enters_step(tmp_path: Path) -> None:
    """Verifies a rejected pre-step checkpoint aborts the turn before dispatch."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("pre-step-failure")

    async def failing_flush(flushed: Any) -> None:
        raise RuntimeError("disk unavailable")

    ctx.on("session/flush", failing_flush)
    mock_llm = ctx.llm.get_provider("mock")

    with pytest.raises(RuntimeError, match="disk unavailable"):
        await ctx.agent_loop.run_turn(session, "Hello agent")

    assert mock_llm is not None
    assert mock_llm.call_history == []


@pytest.mark.asyncio
async def test_turn_end_performs_no_flush(tmp_path: Path) -> None:
    """Verifies a full turn flushes at pre-step only, never at turn/end."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("turn-end")
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))

    reply = await ctx.agent_loop.run_turn(session, "Hello agent")

    assert reply == "Default mock response"
    # Exactly one flush: the pre-step checkpoint. The turn/end boundary
    # adds none, and the non-streaming model call carries no checkpoint.
    assert flushes == [session.session_id]
    ctx.emit("turn/end", session, "turn-1")
    assert flushes == [session.session_id]


@pytest.mark.asyncio
async def test_stream_turn_flushes_at_pre_step_and_llm_stream(tmp_path: Path) -> None:
    """Verifies a streaming turn checkpoints at pre-step and before dispatch."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    session = ctx.sessions.create("stream-turn")
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))

    yielded = []
    async for event in ctx.agent_loop.stream_turn(session, "Hello agent"):
        yielded.append(event)

    assert any(event["type"] == "chunk" for event in yielded)
    assert flushes == [session.session_id, session.session_id]


@pytest.mark.asyncio
async def test_policy_dispose_removes_all_wrappers(tmp_path: Path) -> None:
    """Verifies disposing the policy stops checkpointing on every point."""
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()
    order: list[str] = []
    provider = _install_recording_provider(ctx, order)
    session = ctx.sessions.create("disposed-policy")
    flushes: list[str] = []
    ctx.on("session/flush", lambda s: flushes.append(s.session_id))

    chunks = await _drain(ctx.llm.stream(messages=[], session_id=session.session_id))
    assert flushes == [session.session_id]
    assert provider.calls == 1
    assert [chunk.text for chunk in chunks] == ["hello"]

    # Disabling the row unloads the wiring (and is idempotent).
    await ctx.loader.disable("session-checkpoint-policy")
    await ctx.loader.disable("session-checkpoint-policy")
    await ctx.flush()

    chunks = await _drain(ctx.llm.stream(messages=[], session_id=session.session_id))
    assert flushes == [session.session_id]
    assert provider.calls == 2
    assert [chunk.text for chunk in chunks] == ["hello"]

    decision = await ctx.waterfall(
        "agent/pre-step",
        {"action": "continue", "step_index": 1, "session": session},
        default_handler=lambda decision: decision,
    )
    assert decision["action"] == "continue"
    assert flushes == [session.session_id]
