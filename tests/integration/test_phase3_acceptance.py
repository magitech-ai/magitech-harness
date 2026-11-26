"""Phase 3 end-to-end acceptance and replay conformance suite."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from animas import EventMode

from magitech_harness import check_bundle_entries, create_harness
from magitech_harness.agent.cancellation import AbortController, AbortSignal
from magitech_harness.fs.types import FsError
from magitech_harness.jobs.types import JobOutputSource, JobOutputSourceRead
from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.listing_cache import ModelListingCache
from magitech_harness.llm.listing_config import ListingSettings
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.retry import apply as apply_llm_retry
from magitech_harness.llm.service import LlmAdapter
from magitech_harness.llm.types import (
    GenerateOptions,
    LlmModelInfo,
    StreamChunk,
)
from magitech_harness.llm.values import is_frozen
from magitech_harness.session.events import EventType
from magitech_harness.session.repair import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
)
from magitech_harness.tools import define_tool
from magitech_harness.tools.contracts import (
    ToolBatchInput,
    ToolBatchResult,
    ToolExecutor,
    ToolModelInput,
)


def _make_text_adapter(
    provider_id: str,
    on_stream: Callable[[GenerateOptions], AsyncIterator[StreamChunk]],
) -> LlmAdapter:
    """Build a test LlmAdapter wrapping an async generator function."""

    class _FnAdapter(LlmAdapter):
        async def list_models(self, provider: str) -> list[LlmModelInfo]:
            return [{"id": "m1", "name": "M1", "provider": provider}]

        async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
            async for chunk in on_stream(options):
                yield chunk

    del provider_id
    return _FnAdapter()


def _message_texts(messages: Any) -> list[str]:
    """Extract concatenated text snippets from derived request messages."""
    texts: list[str] = []
    for msg in messages:
        for block in msg.get("content", []):
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
    return texts


@pytest.mark.asyncio
async def test_phase3_scripted_transcript_replay_and_resume(tmp_path: Path) -> None:
    """Replays scripted turns, checks GenerateOptions, reloads, and resumes Turn 3."""
    ctx1 = create_harness(storage_dir=tmp_path)
    await ctx1.flush()
    check_bundle_entries(ctx1)
    loader = ctx1.get("loader", None, strict=False)
    assert loader is not None
    await loader.disable("session-title-first-prompt")

    def add_nums(args: dict[str, Any], _run_ctx: Any) -> dict[str, int]:
        return {"sum": int(args["a"]) + int(args["b"])}

    ctx1.tools.register(
        define_tool(
            {
                "name": "add_nums",
                "description": "Add two integers.",
                "parameters": {
                    "a": {"type": "integer", "required": True},
                    "b": {"type": "integer", "required": True},
                },
                "isConcurrencySafe": lambda _args: True,
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": lambda _args, val: [
                        {"type": "text", "text": json.dumps(val)}
                    ],
                },
                "execute": add_nums,
            }
        ),
        {"when": lambda _c: True},
    )

    step_calls = 0
    captured_options: list[GenerateOptions] = []

    async def _script(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        nonlocal step_calls
        captured_options.append(options)
        assert is_frozen(options)
        step_calls += 1
        if step_calls == 1:
            yield {"type": "block-start", "index": 0, "blockType": "reasoning"}
            yield {"type": "reasoning-delta", "index": 0, "text": "Need sum first."}
            yield {
                "type": "block-end",
                "index": 0,
                "block": {"type": "reasoning", "text": "Need sum first."},
            }
            yield {
                "type": "tool-call-delta",
                "index": 1,
                "id": "call_add_1",
                "name": "add_nums",
                "argumentsDelta": '{"a": 20, "b": 22}',
            }
            yield {
                "type": "usage",
                "usage": {"inputTokens": 100, "outputTokens": 12, "totalTokens": 112},
            }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
            return
        if step_calls == 2:
            yield {"type": "block-start", "index": 0, "blockType": "text"}
            yield {"type": "text-delta", "index": 0, "text": "Total "}
            yield {"type": "text-delta", "index": 0, "text": "is 42."}
            yield {
                "type": "block-end",
                "index": 0,
                "block": {"type": "text", "text": "Total is 42."},
            }
            yield {
                "type": "usage",
                "usage": {"inputTokens": 140, "outputTokens": 10, "totalTokens": 150},
            }
            yield {"type": "finish", "reason": {"kind": "stop"}}
            return
        yield {"type": "text-delta", "index": 0, "text": "Turn 2 complete."}
        yield {
            "type": "usage",
            "usage": {"inputTokens": 50, "outputTokens": 5, "totalTokens": 55},
        }
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx1.llm.register_adapter(["replay"], _make_text_adapter("replay", _script))

    handle1 = await ctx1.agents.create(
        {
            "sessionId": "replay-sess",
            "meta": {"cwd": str(tmp_path)},
            "agentOptions": {"provider": "replay", "model": "m1"},
        }
    )
    agent1 = handle1["agent"]
    agent1.followup(
        create_user_message(
            content=[{"type": "text", "text": "Compute 20 + 22"}],
            source={"kind": "user"},
        )
    )
    await agent1.when_idle()

    agent1.followup(
        create_user_message(
            content=[{"type": "text", "text": "Confirm done"}],
            source={"kind": "user"},
        )
    )
    await agent1.when_idle()

    assert len(captured_options) == 3
    assert [t["name"] for t in captured_options[0]["tools"]] == [
        "add_nums",
        "ask_user_question",
        "bash",
        "create_goal",
        "edit_file",
        "get_goal",
        "glob",
        "grep",
        "interrupt_agent",
        "job_kill",
        "job_list",
        "job_output",
        "list_agents",
        "list_subagent_models",
        "plan",
        "read_file",
        "send_message",
        "subagent",
        "subagent_fork",
        "todo_write",
        "update_goal",
        "wait_agent",
        "web_fetch",
        "web_search",
        "write_file",
    ]
    assert captured_options[0]["sessionId"] == "replay-sess"
    assert "Compute 20 + 22" in _message_texts(captured_options[0]["messages"])
    assert any(msg.get("role") == "tool" for msg in captured_options[1]["messages"])
    assert "Confirm done" in _message_texts(captured_options[2]["messages"])

    trace_types = [
        ev.type
        for ev in agent1.session.events
        if ev.type
        in {
            EventType.TURN_START,
            EventType.USER_MESSAGE,
            EventType.STEP_START,
            EventType.REQUEST_HEADER,
            EventType.REQUEST_CONTEXT,
            EventType.ASSISTANT_MESSAGE,
            EventType.TOOL_CALL,
            EventType.TOOL_RESULT,
            EventType.STEP_END,
            EventType.TURN_END,
        }
    ]
    assert trace_types == [
        EventType.TURN_START,
        EventType.STEP_START,
        EventType.USER_MESSAGE,
        EventType.USER_MESSAGE,
        EventType.USER_MESSAGE,
        EventType.REQUEST_HEADER,
        EventType.REQUEST_CONTEXT,
        EventType.ASSISTANT_MESSAGE,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.STEP_END,
        EventType.STEP_START,
        EventType.ASSISTANT_MESSAGE,
        EventType.STEP_END,
        EventType.TURN_END,
        EventType.TURN_START,
        EventType.STEP_START,
        EventType.USER_MESSAGE,
        EventType.ASSISTANT_MESSAGE,
        EventType.STEP_END,
        EventType.TURN_END,
    ]
    tool_result_ev = next(
        ev for ev in agent1.session.events if ev.type == EventType.TOOL_RESULT
    )
    assert tool_result_ev.payload["message"]["isError"] is False
    assert json.loads(tool_result_ev.payload["message"]["content"][0]["text"]) == {
        "sum": 42
    }
    usage_proj = ctx1.sessionProjections.snapshot(agent1.session)["values"][
        "tokenUsage"
    ]
    assert sum(usage_proj.values()) == 317
    derived_before = agent1.session.derive_messages()

    await handle1["dispose"]()
    await ctx1.fiber.dispose()

    # Reload persisted JSONL and resume Turn 3 in a fresh harness
    ctx2 = create_harness(storage_dir=tmp_path)
    await ctx2.flush()
    loader2 = ctx2.get("loader", None, strict=False)
    assert loader2 is not None
    await loader2.disable("session-title-first-prompt")

    resumed_options: list[GenerateOptions] = []

    async def _turn3_script(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        resumed_options.append(options)
        assert is_frozen(options)
        yield {"type": "text-delta", "index": 0, "text": "Turn 3 resumed."}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx2.llm.register_adapter(["replay"], _make_text_adapter("replay", _turn3_script))
    reloaded = ctx2.sessions.load("replay-sess")
    assert reloaded.derive_messages() == derived_before
    assert (
        ctx2.sessionProjections.snapshot(reloaded)["values"]["tokenUsage"] == usage_proj
    )
    assert ctx2.sessions.close("replay-sess") is True

    handle2 = await ctx2.agents.resume(
        {
            "resumeSessionId": "replay-sess",
            "agentOptions": {"provider": "replay", "model": "m1"},
        }
    )
    agent2 = handle2["agent"]
    agent2.followup(
        create_user_message(
            content=[{"type": "text", "text": "Continue after restart"}],
            source={"kind": "user"},
        )
    )
    await agent2.when_idle()

    assert len(resumed_options) == 1
    resumed_texts = _message_texts(resumed_options[0]["messages"])
    assert "Compute 20 + 22" in resumed_texts
    assert "Turn 2 complete." in resumed_texts
    assert "Continue after restart" in resumed_texts

    await handle2["dispose"]()
    await ctx2.fiber.dispose()


@pytest.mark.asyncio
async def test_phase3_parallel_tool_batch_ten_concurrent_and_serial_barriers(
    tmp_path: Path,
) -> None:
    """Proves bounded (<=10) concurrency for safe calls, ordering, and serial barriers."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    loader = ctx.get("loader", None, strict=False)
    assert loader is not None
    await loader.disable("session-title-first-prompt")

    barrier_10 = asyncio.Barrier(10)
    concurrent_indices: list[int] = []
    serial_log: list[str] = []
    active_safe = 0
    max_active_safe = 0

    async def parallel_probe(args: dict[str, Any], _run_ctx: Any) -> dict[str, Any]:
        idx = int(args["idx"])
        concurrent_indices.append(idx)
        await asyncio.wait_for(barrier_10.wait(), timeout=2.0)
        return {"idx": idx}

    async def safe_ordered(args: dict[str, Any], _run_ctx: Any) -> dict[str, Any]:
        nonlocal active_safe, max_active_safe
        idx = int(args["idx"])
        active_safe += 1
        max_active_safe = max(max_active_safe, active_safe)
        serial_log.append(f"safe-start:{idx}")
        try:
            # Staggered latency to verify call-index ordering despite out-of-order finish
            await asyncio.sleep((12 - idx) * 0.002)
            return {"ordered": idx}
        finally:
            serial_log.append(f"safe-end:{idx}")
            active_safe -= 1

    async def serial_barrier_tool(
        args: dict[str, Any], _run_ctx: Any
    ) -> dict[str, Any]:
        label = str(args["label"])
        serial_log.append(f"start:{label}")
        await asyncio.sleep(0.005)
        serial_log.append(f"end:{label}")
        return {"label": label}

    def _render_json(_args: Any, val: Any) -> list[dict[str, Any]]:
        return [{"type": "text", "text": json.dumps(val)}]

    ctx.tools.register(
        define_tool(
            {
                "name": "parallel_probe",
                "description": "Synchronize 10 calls via barrier.",
                "parameters": {"idx": {"type": "integer", "required": True}},
                "isConcurrencySafe": lambda _args: True,
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": _render_json,
                },
                "execute": parallel_probe,
            }
        ),
        {"when": lambda _c: True},
    )
    ctx.tools.register(
        define_tool(
            {
                "name": "safe_ordered",
                "description": "Concurrency-safe tool with staggered latency.",
                "parameters": {"idx": {"type": "integer", "required": True}},
                "isConcurrencySafe": lambda _args: True,
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": _render_json,
                },
                "execute": safe_ordered,
            }
        ),
        {"when": lambda _c: True},
    )
    ctx.tools.register(
        define_tool(
            {
                "name": "serial_barrier_tool",
                "description": "Non-concurrency-safe barrier tool.",
                "parameters": {"label": {"type": "string", "required": True}},
                "isConcurrencySafe": lambda _args: False,
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": _render_json,
                },
                "execute": serial_barrier_tool,
            }
        ),
        {"when": lambda _c: True},
    )

    step = 0

    async def _batch_script(_opts: GenerateOptions) -> AsyncIterator[StreamChunk]:
        nonlocal step
        step += 1
        if step == 1:
            for i in range(10):
                yield {
                    "type": "tool-call-delta",
                    "index": i,
                    "id": f"p10_{i}",
                    "name": "parallel_probe",
                    "argumentsDelta": json.dumps({"idx": i}),
                }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
            return
        if step == 2:
            yield {
                "type": "tool-call-delta",
                "index": 0,
                "id": "s_a",
                "name": "serial_barrier_tool",
                "argumentsDelta": '{"label": "A"}',
            }
            for i in range(12):
                yield {
                    "type": "tool-call-delta",
                    "index": i + 1,
                    "id": f"o_{i}",
                    "name": "safe_ordered",
                    "argumentsDelta": json.dumps({"idx": i}),
                }
            yield {
                "type": "tool-call-delta",
                "index": 13,
                "id": "s_b",
                "name": "serial_barrier_tool",
                "argumentsDelta": '{"label": "B"}',
            }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
            return
        yield {"type": "text-delta", "index": 0, "text": "Batch complete."}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx.llm.register_adapter(["batch"], _make_text_adapter("batch", _batch_script))
    handle = await ctx.agents.create(
        {
            "sessionId": "batch-sess",
            "meta": {"cwd": str(tmp_path)},
            "agentOptions": {"provider": "batch", "model": "m1"},
        }
    )
    agent = handle["agent"]
    agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Run parallel batches"}],
            source={"kind": "user"},
        )
    )
    await agent.when_idle()

    assert sorted(concurrent_indices) == list(range(10))
    assert 2 <= max_active_safe <= 10
    assert serial_log[:2] == ["start:A", "end:A"]
    assert serial_log[-2:] == ["start:B", "end:B"]
    assert len(serial_log) == 2 + 24 + 2

    result_events = [
        ev for ev in agent.session.events if ev.type == EventType.TOOL_RESULT
    ]
    assert len(result_events) == 24
    p10_results = [
        json.loads(ev.payload["message"]["content"][0]["text"])
        for ev in result_events[:10]
    ]
    assert [item["idx"] for item in p10_results] == list(range(10))
    assert json.loads(result_events[10].payload["message"]["content"][0]["text"]) == {
        "label": "A"
    }
    ordered_results = [
        json.loads(ev.payload["message"]["content"][0]["text"])
        for ev in result_events[11:23]
    ]
    assert [item["ordered"] for item in ordered_results] == list(range(12))
    assert json.loads(result_events[23].payload["message"]["content"][0]["text"]) == {
        "label": "B"
    }

    await handle["dispose"]()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_phase3_approval_fail_closed_in_pipeline(tmp_path: Path) -> None:
    """Proves pre-execute `ask` fails closed across verdicts and live agent turns."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    loader = ctx.get("loader", None, strict=False)
    assert loader is not None
    await loader.disable("session-title-first-prompt")

    handler_calls = 0

    def sensitive_op(_args: dict[str, Any], _run_ctx: Any) -> dict[str, bool]:
        nonlocal handler_calls
        handler_calls += 1
        return {"ran": True}

    ctx.tools.register(
        define_tool(
            {
                "name": "sensitive_op",
                "description": "Sensitive tool.",
                "parameters": {},
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": lambda _args, val: [
                        {"type": "text", "text": json.dumps(val)}
                    ],
                },
                "execute": sensitive_op,
            }
        )
    )
    ctx.on(
        "tools/pre-execute",
        lambda _exec, _nxt: {
            "kind": "ask",
            "reason": "Requires user consent",
            "displayReason": "Confirm sensitive op",
        },
        mode=EventMode.WATERFALL,
    )

    # 1) Default bundle has no `approval` slot -> fail-closed deny
    fake_agent = SimpleNamespace(session=ctx.sessions.create("approval-sess"))
    batch_unmounted = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "c1",
                    "name": "sensitive_op",
                    "arguments": "{}",
                }
            ],
            "signal": AbortController().signal,
            "agent": fake_agent,
        }
    )
    res_unmounted = batch_unmounted["results"][0]["result"]
    assert res_unmounted["isError"] is True
    # The approval service is mounted by default; an ask outside an open
    # turn fails closed with the turn-enclosure error (finding #14).
    assert "outside an open turn" in res_unmounted["content"][0]["text"]
    assert handler_calls == 0

    # 2) Mounted approval returning rejected / unavailable / cancelled & agent=None
    received_reqs: list[dict[str, Any]] = []

    class _StubApproval:
        verdict = "rejected"

        async def request(self, req: dict[str, Any]) -> str:
            await asyncio.sleep(0)
            received_reqs.append(req)
            return self.verdict

    stub = _StubApproval()
    ctx.set("approval", stub)

    batch_no_agent = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "c0",
                    "name": "sensitive_op",
                    "arguments": "{}",
                }
            ],
            "signal": AbortController().signal,
        }
    )
    assert batch_no_agent["results"][0]["result"]["isError"] is True
    assert handler_calls == 0

    for call_id, verdict in (
        ("c2", "rejected"),
        ("c3", "unavailable"),
        ("c4", "cancelled"),
    ):
        stub.verdict = verdict
        batch_denied = await ctx.tools.execute(
            {
                "calls": [
                    {
                        "type": "tool-call",
                        "id": call_id,
                        "name": "sensitive_op",
                        "arguments": "{}",
                    }
                ],
                "signal": AbortController().signal,
                "agent": fake_agent,
            }
        )
        assert batch_denied["results"][0]["result"]["isError"] is True
        assert handler_calls == 0

    # 3) Explicit allowed-once authorizes single execution in a live ReactLoopAgent turn
    stub.verdict = "allowed-once"
    turn_step = 0

    async def _approval_script(_opts: GenerateOptions) -> AsyncIterator[StreamChunk]:
        nonlocal turn_step
        turn_step += 1
        if turn_step == 1:
            yield {
                "type": "tool-call-delta",
                "index": 0,
                "id": "c_live",
                "name": "sensitive_op",
                "argumentsDelta": "{}",
            }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
            return
        yield {"type": "text-delta", "index": 0, "text": "Approved and executed."}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx.llm.register_adapter(["appr"], _make_text_adapter("appr", _approval_script))
    handle = await ctx.agents.create(
        {
            "sessionId": "appr-live-sess",
            "meta": {"cwd": str(tmp_path)},
            "agentOptions": {"provider": "appr", "model": "m1"},
        }
    )
    live_agent = handle["agent"]
    live_agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Run approved tool"}],
            source={"kind": "user"},
        )
    )
    await live_agent.when_idle()

    assert handler_calls == 1
    live_req = received_reqs[-1]
    assert live_req["agent"] is live_agent
    assert live_req["toolName"] == "sensitive_op"
    assert live_req["callId"] == "c_live"
    assert live_req["reason"] == "Requires user consent"
    assert live_req["displayReason"] == "Confirm sensitive op"
    assert isinstance(live_req["signal"], AbortSignal)

    await handle["dispose"]()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_phase3_inbox_steer_maintenance_turn_stopping_and_no_step_cap(
    tmp_path: Path,
) -> None:
    """Verifies steer, inject, status events, maintenance latch, turn-stopping, and >12 steps."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    loader = ctx.get("loader", None, strict=False)
    assert loader is not None
    await loader.disable("session-title-first-prompt")

    def ping(_args: dict[str, Any], _run_ctx: Any) -> dict[str, str]:
        return {"pong": "ok"}

    ctx.tools.register(
        define_tool(
            {
                "name": "ping",
                "description": "Ping tool.",
                "parameters": {},
                "isConcurrencySafe": lambda _args: True,
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": lambda _args, val: [
                        {"type": "text", "text": json.dumps(val)}
                    ],
                },
                "execute": ping,
            }
        ),
        {"when": lambda _c: True},
    )

    step = 0
    step_texts: list[list[str]] = []
    step1_entered = asyncio.Event()
    allow_step1_finish = asyncio.Event()
    steered_inside_turn_stopping = False

    async def _script(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        nonlocal step
        step += 1
        step_texts.append(_message_texts(options["messages"]))
        if step == 1:
            step1_entered.set()
            await allow_step1_finish.wait()
            yield {
                "type": "tool-call-delta",
                "index": 0,
                "id": "call_p_1",
                "name": "ping",
                "argumentsDelta": "{}",
            }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
            return
        if step <= 13:
            # Exceed legacy max_steps=10 in one turn
            yield {
                "type": "tool-call-delta",
                "index": 0,
                "id": f"call_p_{step}",
                "name": "ping",
                "argumentsDelta": "{}",
            }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
            return
        if step == 14:
            yield {"type": "text-delta", "index": 0, "text": "Hit max tokens."}
            yield {"type": "finish", "reason": {"kind": "max-tokens"}}
            return
        yield {"type": "text-delta", "index": 0, "text": "Continued after max-tokens."}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx.llm.register_adapter(["inbox"], _make_text_adapter("inbox", _script))
    handle = await ctx.agents.create(
        {
            "sessionId": "inbox-sess",
            "meta": {"cwd": str(tmp_path)},
            "agentOptions": {"provider": "inbox", "model": "m1"},
        }
    )
    agent = handle["agent"]

    statuses: list[str] = []
    stop_status = ctx.on(
        "agent/status",
        lambda p: statuses.append(str(p.get("status"))),
    )

    def _on_turn_stopping(_payload: dict[str, Any]) -> None:
        nonlocal steered_inside_turn_stopping
        if not steered_inside_turn_stopping:
            steered_inside_turn_stopping = True
            agent.steer(
                create_user_message(
                    content=[{"type": "text", "text": "One more clarification."}],
                    source={"kind": "user"},
                )
            )

    stop_ts = ctx.on("agent/turn-stopping", _on_turn_stopping)

    # Non-waking inject stays idle until next followup wakes the driver
    agent.inject(
        create_user_message(
            content=[{"type": "text", "text": "Injected reminder."}],
            source={"kind": "user"},
        )
    )
    await asyncio.sleep(0.02)
    assert agent.status == "idle"
    assert step == 0

    # 1) Steer while step 1 is in-flight + verify >12 steps + sticky max-tokens
    agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Initial turn"}],
            source={"kind": "user"},
        )
    )
    await asyncio.wait_for(step1_entered.wait(), timeout=5.0)
    agent.steer(
        create_user_message(
            content=[{"type": "text", "text": "Mid-step steer"}],
            source={"kind": "user"},
        )
    )
    allow_step1_finish.set()
    await asyncio.wait_for(agent.when_idle(), timeout=5.0)

    assert step == 15
    assert "Injected reminder." in step_texts[0]
    assert "Mid-step steer" not in step_texts[0]
    assert "Mid-step steer" in step_texts[1]
    assert "One more clarification." in step_texts[14]
    turn_starts = [ev for ev in agent.session.events if ev.type == EventType.TURN_START]
    turn_ends = [ev for ev in agent.session.events if ev.type == EventType.TURN_END]
    assert len(turn_starts) == 1
    assert len(turn_ends) == 1
    assert turn_ends[0].payload["reason"] == {"kind": "max-tokens"}
    assert "running" in statuses and statuses[-1] == "idle"

    # 2) Steer during maintenance latches & replays cleanly, public status stays "idle"
    in_maintenance = asyncio.Event()
    release_maintenance = asyncio.Event()

    async def _maint(_signal: Any) -> str:
        assert agent.status == "idle"
        with pytest.raises(RuntimeError, match="already has active work"):
            await agent.run_maintenance(_maint)
        in_maintenance.set()
        await asyncio.wait_for(release_maintenance.wait(), timeout=5.0)
        return "maint-ok"

    maint_task = asyncio.create_task(agent.run_maintenance(_maint))
    await asyncio.wait_for(in_maintenance.wait(), timeout=5.0)
    agent.steer(
        create_user_message(
            content=[{"type": "text", "text": "Latched during maintenance"}],
            source={"kind": "user"},
        )
    )
    release_maintenance.set()
    assert await asyncio.wait_for(maint_task, timeout=5.0) == "maint-ok"
    await asyncio.wait_for(agent.when_idle(), timeout=5.0)
    assert len([e for e in agent.session.events if e.type == EventType.TURN_END]) == 2
    assert "Latched during maintenance" in step_texts[-1]

    # 3) pre-step reject closes turn as blocked with ZERO user/message or step/start events
    reject_once = False

    async def _pre_step_reject(payload: dict[str, Any], nxt: Callable[..., Any]) -> Any:
        nonlocal reject_once
        if not reject_once and payload.get("agent") is agent:
            reject_once = True
            return {"kind": "reject"}
        return await nxt(payload)

    stop_pre = ctx.on("agent/pre-step", _pre_step_reject, mode=EventMode.WATERFALL)
    before_steps = len(
        [ev for ev in agent.session.events if ev.type == EventType.STEP_START]
    )
    before_users = len(
        [ev for ev in agent.session.events if ev.type == EventType.USER_MESSAGE]
    )
    before_llm_calls = step
    agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Blocked turn"}],
            source={"kind": "user"},
        )
    )
    await asyncio.wait_for(agent.when_idle(), timeout=5.0)
    assert (
        len([ev for ev in agent.session.events if ev.type == EventType.STEP_START])
        == before_steps
    )
    assert (
        len([ev for ev in agent.session.events if ev.type == EventType.USER_MESSAGE])
        == before_users
    )
    assert step == before_llm_calls
    blocked_end = [ev for ev in agent.session.events if ev.type == EventType.TURN_END][
        -1
    ]
    assert blocked_end.payload["reason"] == {"kind": "blocked"}

    stop_status()
    stop_pre()
    stop_ts()
    await handle["dispose"]()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_phase3_retry_rate_limit_five_retries_and_model_listing_cache(
    tmp_path: Path,
) -> None:
    """Proves 5 rate-limit retries, exhaustion, and ModelListingCache rescope."""
    recorded_delays: list[float] = []

    async def _fast_sleep(delay_ms: float, _signal: AbortSignal) -> None:
        recorded_delays.append(delay_ms)

    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    loader = ctx.get("loader", None, strict=False)
    assert loader is not None
    await loader.disable("session-title-first-prompt")
    await loader.disable("llm-retry")
    await apply_llm_retry(
        ctx,
        internals={"sleep": _fast_sleep, "random": lambda: 0.5},
    )

    attempts = 0
    exhaust_always = False

    async def _retry_script(_opts: GenerateOptions) -> AsyncIterator[StreamChunk]:
        nonlocal attempts
        attempts += 1
        if exhaust_always or attempts <= 5:
            raise LlmError("rate limited", "RATE_LIMIT", status=429)
        yield {"type": "text-delta", "index": 0, "text": "Recovered on attempt 6."}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx.llm.register_adapter(["rlim"], _make_text_adapter("rlim", _retry_script))
    handle = await ctx.agents.create(
        {
            "sessionId": "retry-sess",
            "meta": {"cwd": str(tmp_path)},
            "agentOptions": {"provider": "rlim", "model": "m1"},
        }
    )
    agent = handle["agent"]
    agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Retry test"}],
            source={"kind": "user"},
        )
    )
    await agent.when_idle()

    assert attempts == 6
    assert recorded_delays == [500, 1000, 2000, 4000, 8000]
    retry_events = [ev for ev in agent.session.events if ev.type == EventType.LLM_RETRY]
    retry_started = [
        ev for ev in agent.session.events if ev.type == EventType.LLM_RETRY_STARTED
    ]
    assert len(retry_events) == 5
    assert len(retry_started) == 5
    assert [ev.payload["retry"] for ev in retry_events] == [1, 2, 3, 4, 5]
    assert all(ev.payload["maxRetries"] == 5 for ev in retry_events)
    assert [ev.payload["delayMs"] for ev in retry_events] == [
        500,
        1000,
        2000,
        4000,
        8000,
    ]
    assert retry_events[0].payload["failure"]["code"] == "RATE_LIMIT"

    # Retry exhaustion: 6th consecutive failure terminates turn with error
    exhaust_always = True
    agent.followup(
        create_user_message(
            content=[{"type": "text", "text": "Exhaust retries"}],
            source={"kind": "user"},
        )
    )
    await agent.when_idle()
    last_turn_end = [
        ev for ev in agent.session.events if ev.type == EventType.TURN_END
    ][-1]
    assert last_turn_end.payload["reason"]["kind"] == "error"

    # ModelListingCache: single-flight cold load + credential-rescope + models-updated
    loads = 0
    cred_holder = ["secret-1"]
    notified: list[str] = []
    ctx.on("llm/models-updated", lambda: notified.append("updated"))
    settings: ListingSettings = {
        "url": "https://api.example.com/models",
        "timeoutMs": 5_000,
        "maxAgeMs": 600_000,
    }

    async def _fetch_models(_url: str, cred: str, _sig: AbortSignal) -> dict[str, Any]:
        nonlocal loads
        loads += 1
        await asyncio.sleep(0.01)
        model_id = "live-m1" if cred == "secret-1" else "live-m2"
        return {
            "entries": [{"id": model_id, "name": model_id, "provider": "x"}],
            "skipped": 0,
        }

    cache = ModelListingCache(
        {
            "settings": lambda: settings,
            "credential": lambda _url: cred_holder[0],
            "load": _fetch_models,
            "now": lambda: 0.0,
            "info": lambda _msg: None,
            "warn": lambda _msg: None,
            "changed": lambda: ctx.emit("llm/models-updated"),
        }
    )

    view_a, view_b = await asyncio.gather(cache.view(settings), cache.view(settings))
    assert loads == 1
    assert view_a["authenticated"] is True
    listing_a = cast(dict[str, Any] | None, view_a["listing"])
    listing_b = cast(dict[str, Any] | None, view_b["listing"])
    assert listing_a is not None and listing_b is not None
    assert [m["id"] for m in listing_a["entries"]] == ["live-m1"]
    assert [m["id"] for m in listing_b["entries"]] == ["live-m1"]
    assert notified == ["updated"]

    # Credential rescope forces a fresh fetch for secret-2
    cred_holder[0] = "secret-2"
    view_c = await cache.view(settings)
    listing_c = cast(dict[str, Any] | None, view_c["listing"])
    assert loads == 2
    assert listing_c is not None
    assert [m["id"] for m in listing_c["entries"]] == ["live-m2"]
    assert len(notified) == 2

    await cache.dispose()
    await handle["dispose"]()
    await ctx.fiber.dispose()


@pytest.mark.asyncio
async def test_phase3_crash_recovery_def04_and_per_step_tool_recovery(
    tmp_path: Path,
) -> None:
    """Proves DEF-04 startup closers (started+unstarted) and live ToolCallRecovery."""
    # 1) Persist a crashed unclosed turn + step + started (orphan_1) & unstarted (orphan_2)
    ctx1 = create_harness(storage_dir=tmp_path)
    await ctx1.flush()
    s1 = ctx1.sessions.create("crash-sess")
    ctx1.sessions.log_event(
        s1,
        EventType.TURN_START,
        {"turn": 1, "inputIds": ["in-1"]},
    )
    ctx1.sessions.log_event(
        s1,
        EventType.STEP_START,
        {"turn": 1, "step": 1, "consumedInputIds": ["in-1"]},
    )
    ctx1.sessions.log_event(
        s1,
        EventType.ASSISTANT_MESSAGE,
        {
            "turn": 1,
            "step": 1,
            "stream": [],
            "message": {
                "id": "asst-crash-1",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool-call",
                        "id": "orphan_1",
                        "name": "noop",
                        "arguments": "{}",
                    },
                    {
                        "type": "tool-call",
                        "id": "orphan_2",
                        "name": "noop",
                        "arguments": "{}",
                    },
                ],
                "source": {"kind": "model", "provider": "rec", "model": "m1"},
            },
        },
    )
    call_ev = ctx1.sessions.log_event(
        s1,
        EventType.TOOL_CALL,
        {"turn": 1, "step": 1, "callId": "orphan_1", "name": "noop", "arguments": {}},
    )
    assert call_ev is not None
    await ctx1.sessions.flush(s1)
    assert ctx1.sessions.close("crash-sess") is True
    await ctx1.fiber.dispose()

    # 2) Reboot & resume in a fresh harness -> repairs with synthetic interrupted closers
    ctx2 = create_harness(storage_dir=tmp_path)
    await ctx2.flush()
    loader2 = ctx2.get("loader", None, strict=False)
    assert loader2 is not None
    await loader2.disable("session-title-first-prompt")

    def noop(_args: dict[str, Any], _run_ctx: Any) -> dict[str, bool]:
        return {"ok": True}

    ctx2.tools.register(
        define_tool(
            {
                "name": "noop",
                "description": "Noop tool.",
                "parameters": {},
                "isConcurrencySafe": lambda _args: False,
                "output": {
                    "schema": {"type": "object", "additionalProperties": True},
                    "render": lambda _args, val: [
                        {"type": "text", "text": json.dumps(val)}
                    ],
                },
                "execute": noop,
            }
        ),
        {"when": lambda _c: True},
    )

    async def _recovery_script(_opts: GenerateOptions) -> AsyncIterator[StreamChunk]:
        yield {
            "type": "tool-call-delta",
            "index": 0,
            "id": "live_started",
            "name": "noop",
            "argumentsDelta": "{}",
        }
        yield {
            "type": "tool-call-delta",
            "index": 1,
            "id": "live_unstarted",
            "name": "noop",
            "argumentsDelta": "{}",
        }
        yield {"type": "finish", "reason": {"kind": "tool-calls"}}

    ctx2.llm.register_adapter(["rec"], _make_text_adapter("rec", _recovery_script))
    handle2 = await ctx2.agents.resume(
        {
            "resumeSessionId": "crash-sess",
            "agentOptions": {"provider": "rec", "model": "m1"},
        }
    )
    agent2 = handle2["agent"]
    repaired_tail = agent2.session.events[4:]
    assert [ev.type for ev in repaired_tail] == [
        EventType.TOOL_RESULT,
        EventType.TOOL_RESULT,
        EventType.STEP_END,
        EventType.TURN_END,
    ]
    assert repaired_tail[0].payload["error"]["code"] == TOOL_OUTCOME_UNKNOWN
    assert repaired_tail[0].source_event_seqs == [call_ev.seq]
    assert repaired_tail[1].payload["error"]["code"] == TOOL_NOT_STARTED
    assert repaired_tail[1].source_event_seqs is None
    assert repaired_tail[3].payload["reason"] == {"kind": "interrupted"}
    await ctx2.sessions.flush(agent2.session)
    jsonl_file = next(tmp_path.rglob("session.v4.jsonl"))
    persisted_types = [
        json.loads(line).get("type")
        for line in jsonl_file.read_text(encoding="utf-8").splitlines()
    ]
    assert "turn/end" in persisted_types and "tool/result" in persisted_types

    # 3) Live per-step ToolCallRecovery when a batch crashes mid-execution
    class _CrashingExecutor:
        def describe(self, definitions: Any) -> ToolModelInput:
            return {"schemas": [], "knownNames": [d["name"] for d in definitions]}

        async def execute(self, batch: ToolBatchInput) -> ToolBatchResult:
            started_call = batch["calls"][0]
            log_info = batch["log"]
            agent2.session.append(
                EventType.TOOL_CALL,
                {
                    "turn": log_info["turn"],
                    "step": log_info["step"],
                    "callId": started_call["id"],
                    "name": started_call["name"],
                    "arguments": {},
                },
            )
            raise RuntimeError("mid-batch executor crash")

    await loader2.disable("tool-executor-native")
    dispose_exec = ctx2.tools.set_executor(cast(ToolExecutor, _CrashingExecutor()))
    agent2.followup(
        create_user_message(
            content=[{"type": "text", "text": "Trigger mid-batch recovery"}],
            source={"kind": "user"},
        )
    )
    await agent2.when_idle()

    turn2_events = agent2.session.events[8:]
    turn2_results = [ev for ev in turn2_events if ev.type == EventType.TOOL_RESULT]
    assert len(turn2_results) == 2
    assert turn2_results[0].payload["error"]["code"] == TOOL_OUTCOME_UNKNOWN
    assert turn2_results[1].payload["error"]["code"] == TOOL_NOT_STARTED
    turn2_end = next(ev for ev in turn2_events if ev.type == EventType.TURN_END)
    assert turn2_end.payload["reason"]["kind"] == "error"

    dispose_exec()
    await handle2["dispose"]()
    await ctx2.fiber.dispose()


def test_phase3_headless_cli_subprocess_exit_codes_and_title_drain(
    tmp_path: Path,
) -> None:
    """Drives real `mth` CLI subprocess for exit-code mapping, --json error, and K5 drain."""
    plugin_mod = tmp_path / "cli_accept_plugins.py"
    plugin_mod.write_text(
        """
from __future__ import annotations
import asyncio
from typing import Any, ClassVar
from magitech_harness.llm.service import LlmAdapter
from magitech_harness.llm.types import ModelResponse

class _AcceptOkAdapter(LlmAdapter):
    async def list_models(self, provider: str) -> list[dict[str, Any]]:
        return [{"id": "m1", "name": "M1", "provider": provider}]

    async def stream(self, options: dict[str, Any]):
        del options
        yield {"type": "block-start", "index": 0, "blockType": "text"}
        yield {"type": "text-delta", "index": 0, "text": "CLI headless ok."}
        yield {
            "type": "block-end",
            "index": 0,
            "block": {"type": "text", "text": "CLI headless ok."},
        }
        yield {"type": "finish", "reason": {"kind": "stop"}}

class _SlowTitleProvider:
    async def generate(self, messages, tools=None, system_prompt=None, **kwargs):
        del messages, tools, system_prompt, kwargs
        await asyncio.sleep(0.08)
        return ModelResponse(text="Async CLI Title")

class _AcceptErrAdapter(LlmAdapter):
    async def list_models(self, provider: str) -> list[dict[str, Any]]:
        return [{"id": "m1", "name": "M1", "provider": provider}]

    async def stream(self, options: dict[str, Any]):
        del options
        raise RuntimeError("cli simulated error")
        yield {"type": "finish", "reason": {"kind": "stop"}}

class OkPlugin:
    inject: ClassVar[list[str]] = ["llm"]
    def __init__(self, ctx: Any) -> None:
        ctx.llm.register_adapter(["okcli"], _AcceptOkAdapter())
        ctx.llm.register_provider("okcli", _SlowTitleProvider())

class ErrPlugin:
    inject: ClassVar[list[str]] = ["llm"]
    def __init__(self, ctx: Any) -> None:
        ctx.llm.register_adapter(["errcli"], _AcceptErrAdapter())
""",
        encoding="utf-8",
    )
    ok_profile = tmp_path / "ok.yaml"
    ok_profile.write_text("['cli_accept_plugins:OkPlugin']", encoding="utf-8")
    err_profile = tmp_path / "err.yaml"
    err_profile.write_text("['cli_accept_plugins:ErrPlugin']", encoding="utf-8")

    storage = tmp_path / "cli_storage"
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{tmp_path}{os.pathsep}{existing_pp}" if existing_pp else str(tmp_path)
    )

    proc_ok = subprocess.run(
        [
            sys.executable,
            "-m",
            "magitech_harness.cli",
            "--storage",
            str(storage),
            "--profile",
            str(ok_profile),
            "--model",
            "okcli/m1",
            "headless",
            "--json",
            "Summarize quarterly launch readiness",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert proc_ok.returncode == 0, proc_ok.stderr
    payload_ok = json.loads(proc_ok.stdout.strip().splitlines()[-1])
    assert payload_ok["exitCode"] == 0
    assert payload_ok["text"] == "CLI headless ok."
    assert payload_ok["reason"] == {"kind": "completed"}

    # Verify K5: slow async provider title landed in persisted JSONL before CLI exited
    session_file = next(storage.rglob("session.v4.jsonl"))
    records = [
        json.loads(line)
        for line in session_file.read_text(encoding="utf-8").splitlines()
    ]
    title_records = [rec for rec in records if rec.get("type") == "session/title"]
    assert title_records
    assert any(
        rec["data"].get("title") == "Async CLI Title"
        and rec["data"].get("source", {}).get("kind") == "provider"
        for rec in title_records
    )

    # Resuming completed session with failing turn returns exit code 1 + JSON error
    proc_err = subprocess.run(
        [
            sys.executable,
            "-m",
            "magitech_harness.cli",
            "--storage",
            str(storage),
            "--profile",
            str(err_profile),
            "--model",
            "errcli/m1",
            "headless",
            "--json",
            "--session-id",
            payload_ok["sessionId"],
            "Fail second turn",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
        check=False,
    )
    assert proc_err.returncode == 1
    payload_err = json.loads(proc_err.stdout.strip().splitlines()[-1])
    assert payload_err["exitCode"] == 1
    assert payload_err["reason"]["kind"] == "error"


@pytest.mark.asyncio
async def test_phase3_infra_services_e2e_through_create_harness(
    tmp_path: Path,
) -> None:
    """Drives P3-9 fs/patch/obs, workingDirectory, shell, shellEnv, and jobs pump."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    check_bundle_entries(ctx)
    loader = ctx.get("loader", None, strict=False)
    assert loader is not None
    await loader.disable("session-title-first-prompt")

    async def _noop_script(_opts: GenerateOptions) -> AsyncIterator[StreamChunk]:
        yield {"type": "text-delta", "index": 0, "text": "ok"}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx.llm.register_adapter(["infra"], _make_text_adapter("infra", _noop_script))
    handle = await ctx.agents.create(
        {
            "sessionId": "infra-sess",
            "meta": {"cwd": str(tmp_path)},
            "agentOptions": {"provider": "infra", "model": "m1"},
        }
    )
    agent = handle["agent"]

    # fs + patch + observation policy + workingDirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    await ctx.workingDirectory.set(agent, str(subdir))
    assert ctx.workingDirectory.get(agent.session) == str(subdir.resolve())
    wd_events = [
        ev
        for ev in agent.session.events
        if ev.type == EventType.WORKING_DIRECTORY_CHANGE
    ]
    assert wd_events and wd_events[-1].payload["cwd"] == str(subdir.resolve())
    await ctx.workingDirectory.set(agent, str(tmp_path))
    assert ctx.workingDirectory.get(agent.session) == str(tmp_path.resolve())

    target = await ctx.fs.resolve(str(tmp_path / "hello.txt"))
    await ctx.fs.write_text(target, "alpha\nbeta\n")
    with pytest.raises(FsError) as unobs_exc:
        await ctx.waterfall("fs/edit-intent", target, {"agent": agent})
    assert unobs_exc.value.code == "FS_NOT_OBSERVED"
    assert await ctx.fs.read_text(target) == "alpha\nbeta\n"
    stat_info = await ctx.fs.stat(target)
    assert stat_info is not None and "version" in stat_info
    ctx.emit(
        "fs/observed",
        target,
        {"kind": "present", "version": stat_info["version"]},
        {"agent": agent},
    )
    edit_intent = await ctx.waterfall("fs/edit-intent", target, {"agent": agent})
    assert edit_intent == {"version": stat_info["version"]}
    edited = await ctx.fs.edit_text(
        target,
        {"oldString": "beta", "newString": "gamma", "replaceAll": False},
        expected={"version": stat_info["version"]},
    )
    assert edited["after"] == "alpha\ngamma\n"
    with pytest.raises(FsError) as stale_exc:
        await ctx.fs.edit_text(
            target,
            {"oldString": "gamma", "newString": "nope", "replaceAll": False},
            expected={"version": stat_info["version"]},
        )
    assert stale_exc.value.code == "FS_STALE_VERSION"

    patched = await ctx.fs.patch_text(
        target,
        [
            {
                "lineNumber": 1,
                "oldLines": ["gamma"],
                "newLines": ["delta"],
                "isEndOfFile": False,
            }
        ],
        expected={"version": edited["version"]},
    )
    assert patched["after"] == "alpha\ndelta\n"

    # shell + shellEnv (MAGITECH_* prefix)
    resolved_env = ctx.shellEnv.collect({"agent": agent, "cwd": str(tmp_path)})
    assert resolved_env["MAGITECH_SESSION_ID"] == agent.session.id
    cmd = (
        "Write-Output 'shell-e2e-ok'"
        if sys.platform == "win32"
        else "printf 'shell-e2e-ok'"
    )
    shell_spec = ctx.shell.resolve(
        {"command": cmd, "workdir": str(tmp_path), "env": resolved_env}
    )
    exec_handle = await ctx.shell.execute(shell_spec)
    outcome = await exec_handle.result()
    assert outcome["exitCode"] == 0
    assert "shell-e2e-ok" in outcome["stdout"]["text"]

    # jobs + live pump + read/read_at + kill
    loop = asyncio.get_running_loop()
    job_done: asyncio.Future[dict[str, Any]] = loop.create_future()
    chunks = ["live-job-byte\n"]

    class _PullOutputSource:
        @property
        def channel(self) -> str:
            return "stdout"

        def read(self, from_byte: int) -> JobOutputSourceRead:
            data = "".join(chunks).encode("utf-8")
            tail = data[from_byte:].decode("utf-8")
            return {"text": tail, "nextOffset": len(data)}

    def _job_run(_job_handle: Any) -> SimpleNamespace:
        def _cancel(_reason: str | None = None) -> None:
            if not job_done.done():
                job_done.set_result({"status": "killed"})

        return SimpleNamespace(done=job_done, cancel=_cancel)

    dispose_ctrl = ctx.jobs.attach_controller("accept-owner")
    job_id = ctx.jobs.start(
        {
            "kind": "custom",
            "label": "accept-job",
            "run": _job_run,
            "output": [cast(JobOutputSource, _PullOutputSource())],
        }
    )
    for _ in range(40):
        ring_text = "".join(c["text"] for c in ctx.jobs.read_at(job_id, 0)["chunks"])
        if "live-job-byte\n" in ring_text:
            break
        await asyncio.sleep(0.01)
    assert "live-job-byte\n" in "".join(
        c["text"] for c in ctx.jobs.read_at(job_id, 0)["chunks"]
    )
    assert "live-job-byte\n" in "".join(
        c["text"] for c in ctx.jobs.read(job_id)["chunks"]
    )
    assert ctx.jobs.kill(job_id, reason="done") == "requested"
    settled = await ctx.jobs.wait(job_id, timeout_ms=2000)
    assert settled["status"] == "killed"

    dispose_ctrl()
    await handle["dispose"]()
    await ctx.fiber.dispose()
