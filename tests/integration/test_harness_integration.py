from pathlib import Path

import pytest
from anima import EventMode

from magitech_harness import (
    EventType,
    MockLLMProvider,
    ToolCallData,
    create_harness,
    define_tool,
)


@pytest.mark.asyncio
async def test_full_turn_with_tool_call(tmp_path: Path):
    # 1. Boot harness
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()

    # Verify all 6 capability seams are active
    assert ctx.has("sessions")
    assert ctx.has("tools")
    assert ctx.has("system_prompt")
    assert ctx.has("llm")
    assert ctx.has("agent_loop")
    assert ctx.has("shell")

    # 2. Register custom tool
    executed_tools = []

    def multiply(args: dict) -> int:
        a = args.get("a", 0)
        b = args.get("b", 0)
        res = a * b
        executed_tools.append((a, b, res))
        return res

    ctx.tools.register(
        define_tool(
            name="multiply",
            description="Multiplies two numbers",
            parameters={
                "a": {"type": "integer", "required": True},
                "b": {"type": "integer", "required": True},
            },
            execute=multiply,
        )
    )

    # 3. Configure mock LLM responses for multi-step turn:
    # Step 1: LLM returns tool_call "multiply" (a=6, b=7)
    # Step 2: LLM returns final text answer "The answer is 42."
    mock_llm: MockLLMProvider = ctx.llm.active_provider
    mock_llm.queue_response(
        text="Let me calculate that for you.",
        tool_calls=[
            ToolCallData(call_id="call_1", name="multiply", arguments={"a": 6, "b": 7})
        ],
    )
    mock_llm.queue_response(
        text="The answer is 42.",
        tool_calls=[],
    )

    # 4. Run the turn
    session = ctx.sessions.create("test_session_1")
    final_output = await ctx.agent_loop.run_turn(session, "What is 6 * 7?")

    # Verify output & execution
    assert final_output == "The answer is 42."
    assert executed_tools == [(6, 7, 42)]

    # 5. Verify durable session events
    event_types = [ev.type for ev in session.events]
    assert event_types == [
        EventType.TURN_START,
        EventType.USER_MESSAGE,
        EventType.STEP_START,
        EventType.ASSISTANT_MESSAGE,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.STEP_END,
        EventType.STEP_START,
        EventType.ASSISTANT_MESSAGE,
        EventType.STEP_END,
        EventType.TURN_END,
    ]

    # 6. Verify derive_messages deterministic projection
    projected = session.derive_messages()
    assert len(projected) == 4
    assert projected[0]["role"] == "user"
    assert projected[0]["content"] == "What is 6 * 7?"
    assert projected[1]["role"] == "assistant"
    assert len(projected[1]["tool_calls"]) == 1
    assert projected[2]["role"] == "tool"
    assert projected[2]["content"] == "42"
    assert projected[3]["role"] == "assistant"
    assert projected[3]["content"] == "The answer is 42."

    # 7. Verify JSONL file persistence on disk
    assert session.storage_path.exists()
    lines = session.storage_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 11


@pytest.mark.asyncio
async def test_waterfall_guard_pre_step_veto(tmp_path: Path):
    ctx = create_harness(storage_dir=tmp_path, mock_provider=True)
    await ctx.flush()

    # Install a security plugin via waterfall hook on agent/pre-step
    async def security_guard(decision: dict, next_fn):
        if decision.get("step_index") == 1:
            return {"action": "reject", "reason": "blocked by policy"}
        return await next_fn(decision)

    ctx.on("agent/pre-step", security_guard, mode=EventMode.WATERFALL)

    session = ctx.sessions.create("blocked_session")
    res = await ctx.agent_loop.run_turn(session, "Malicious instruction")

    # Since step 1 was rejected, assistant text is empty and 0 steps ran
    assert res == ""
    event_types = [ev.type for ev in session.events]
    assert event_types == [
        EventType.TURN_START,
        EventType.USER_MESSAGE,
        EventType.STEP_START,
        EventType.TURN_END,
    ]


@pytest.mark.asyncio
async def test_shell_service_execution(tmp_path: Path):
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()

    code, stdout, _stderr = await ctx.shell.run("echo 'Magitech Rocks'")
    assert code == 0
    assert "Magitech Rocks" in stdout
