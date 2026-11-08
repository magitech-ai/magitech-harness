"""Unit tests for AgentLoopService."""

import pytest
from anima import Context, EventMode

from magitech_harness.agent.loop import AgentLoopService
from magitech_harness.llm.service import LLMService, MockLLMProvider
from magitech_harness.session.events import EventType, ToolCallData
from magitech_harness.session.service import SessionService


@pytest.mark.asyncio
async def test_agent_loop_single_step_no_tools() -> None:
    """Verifies a 1-step turn when the LLM returns plain text with no tool calls."""
    ctx = Context()
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(AgentLoopService)
    await ctx.flush()

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    mock_llm.queue_response(text="Direct answer")

    session = ctx.sessions.create("sess_1")
    reply = await ctx.agent_loop.run_turn(session, "Hello agent")

    assert reply == "Direct answer"
    types = [e.type for e in session.events]
    assert types == [
        EventType.TURN_START,
        EventType.USER_MESSAGE,
        EventType.STEP_START,
        EventType.ASSISTANT_MESSAGE,
        EventType.STEP_END,
        EventType.TURN_END,
    ]


@pytest.mark.asyncio
async def test_agent_loop_pre_step_reject_short_circuit() -> None:
    """Verifies that pre-step waterfall veto immediately stops the turn."""
    ctx = Context()
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(AgentLoopService)
    await ctx.flush()

    async def guard(decision: dict, next_fn):
        return {"action": "reject"}

    ctx.on("agent/pre-step", guard, mode=EventMode.WATERFALL)

    session = ctx.sessions.create("sess_veto")
    reply = await ctx.agent_loop.run_turn(session, "Prompt")

    assert reply == ""
    types = [e.type for e in session.events]
    assert types == [
        EventType.TURN_START,
        EventType.USER_MESSAGE,
        EventType.STEP_START,
        EventType.TURN_END,
    ]


@pytest.mark.asyncio
async def test_agent_loop_max_steps_termination() -> None:
    """Verifies that reaching max_steps terminates the loop even if tool calls continue."""
    from magitech_harness.tools.definition import define_tool
    from magitech_harness.tools.service import ToolService

    ctx = Context()
    ctx.plugin(SessionService)
    ctx.plugin(LLMService)
    ctx.plugin(ToolService)
    ctx.plugin(AgentLoopService)
    await ctx.flush()

    ctx.tools.register(
        define_tool(name="ping", description="ping", execute=lambda args: "pong")
    )

    mock_llm = MockLLMProvider()
    ctx.llm.register_provider("mock", mock_llm)
    # Queue infinite tool calls
    for i in range(10):
        mock_llm.queue_response(
            text=f"step_{i}",
            tool_calls=[ToolCallData(call_id=f"c_{i}", name="ping", arguments={})],
        )

    session = ctx.sessions.create("sess_max")
    reply = await ctx.agent_loop.run_turn(session, "Run infinite", max_steps=2)

    assert reply == "step_1"
    step_starts = [e for e in session.events if e.type == EventType.STEP_START]
    assert len(step_starts) == 2
