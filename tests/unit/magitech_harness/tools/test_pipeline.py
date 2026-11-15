"""Unit tests for ToolExecutionPipeline and execute_tool_batch."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from anima import Context, EventMode, Service

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.llm.error import HarnessError
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.values import FrozenDict, is_frozen
from magitech_harness.scope import create_scope
from magitech_harness.session.store import Session
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.contracts import (
    TOOL_ABORTED,
    TOOL_ABORTED_BEFORE_DISPATCH,
    ToolBatchExecutionError,
    ToolDefinition,
)
from magitech_harness.tools.native_executor import apply as apply_native_executor
from magitech_harness.tools.pipeline import (
    ToolExecutionPipeline,
    execute_tool_batch,
    fuse_tool_signals,
    parse_arguments,
    render_content_summary,
    tool_error_message,
)
from magitech_harness.tools.schema import define_tool
from magitech_harness.tools.service import ToolsService


def _echo_tool(
    name: str = "echo",
    **overrides: Any,
) -> ToolDefinition:
    spec: dict[str, Any] = {
        "name": name,
        "description": f"tool {name}",
        "parameters": {"text": {"type": "string"}},
        "output": {
            "schema": {"type": "string"},
            "render": lambda _args, val: [{"type": "text", "text": val}],
        },
        "execute": lambda args, _ctx: args.get("text", ""),
    }
    spec.update(overrides)
    return define_tool(spec)


async def _setup() -> Context:
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolsService)
    ctx.use(apply_native_executor)
    await ctx.flush()
    return ctx


def test_fuse_tool_signals_and_error_helpers() -> None:
    """Fused signal propagates existing/live aborts, cleans up, and formats hostile errors."""
    sig1 = AbortSignal()
    sig2 = AbortSignal()
    fused, dispose = fuse_tool_signals(sig1, None, sig2)
    assert not fused.is_aborted
    sig2.abort("from-sig2")
    assert fused.is_aborted
    assert fused.reason == "from-sig2"
    dispose()

    sig3 = AbortSignal()
    sig3.abort("pre-aborted")
    fused_pre, dispose_pre = fuse_tool_signals(AbortSignal(), sig3)
    assert fused_pre.is_aborted
    assert fused_pre.reason == "pre-aborted"
    dispose_pre()

    class Hostile:
        def __str__(self) -> str:
            raise RuntimeError("str boom")

        def __repr__(self) -> str:
            raise RuntimeError("repr boom")

    assert tool_error_message(Hostile()) == "<unprintable thrown value>"
    assert tool_error_message("raw text") == "raw text"
    assert tool_error_message({"message": "obj msg"}) == "obj msg"
    assert render_content_summary([]) == "tool result blocked by post-execute policy"
    assert (
        render_content_summary([{"type": "reasoning", "text": "secret"}])
        == "[reasoning content]"
    )
    assert render_content_summary([{"type": "text", "text": "hi"}]) == "hi"


@pytest.mark.asyncio
async def test_execution_mode_classification_fails_closed() -> None:
    """execution_mode returns parallel only for strict True on valid args."""
    ctx = await _setup()
    ctx.tools.register(
        _echo_tool(
            "rw",
            parameters={"mode": {"type": "string", "required": True}},
            isConcurrencySafe=lambda args: args.get("mode") == "read",
        )
    )
    ctx.tools.register(_echo_tool("plain"))

    raw_truthy: ToolDefinition = {
        "name": "truthy",
        "description": "returns string",
        "parameters": {"type": "object", "properties": {}},
        "output": {"schema": {"type": "null"}, "render": lambda _a, _v: []},
        "isConcurrencySafe": lambda _a: "yes",  # type: ignore[return-value]
        "execute": lambda _a, _c: None,
    }
    ctx.tools.register(raw_truthy)

    def _boom(_args: Any) -> bool:
        raise RuntimeError("boom")

    ctx.tools.register(
        {
            "name": "thrower",
            "description": "throws",
            "parameters": {"type": "object", "properties": {}},
            "output": {"schema": {"type": "null"}, "render": lambda _a, _v: []},
            "isConcurrencySafe": _boom,
            "execute": lambda _a, _c: None,
        }
    )

    pipeline = ToolExecutionPipeline(ctx)
    sig = AbortSignal()
    assert pipeline.execution_mode(
        {"callId": "1", "name": "rw", "arguments": {"mode": "read"}, "signal": sig}
    ) == {"kind": "parallel"}
    assert pipeline.execution_mode(
        {"callId": "2", "name": "rw", "arguments": {"mode": "write"}, "signal": sig}
    ) == {"kind": "exclusive"}
    assert pipeline.execution_mode(
        {"callId": "3", "name": "rw", "arguments": {}, "signal": sig}
    ) == {"kind": "exclusive"}
    assert pipeline.execution_mode(
        {"callId": "4", "name": "plain", "arguments": {}, "signal": sig}
    ) == {"kind": "exclusive"}
    assert pipeline.execution_mode(
        {"callId": "5", "name": "missing", "arguments": {}, "signal": sig}
    ) == {"kind": "exclusive"}
    assert pipeline.execution_mode(
        {"callId": "6", "name": "truthy", "arguments": {}, "signal": sig}
    ) == {"kind": "exclusive"}
    assert pipeline.execution_mode(
        {"callId": "7", "name": "thrower", "arguments": {}, "signal": sig}
    ) == {"kind": "exclusive"}


@pytest.mark.asyncio
async def test_pipeline_unknown_error_timeout_and_hostile_throw() -> None:
    """Unknown tool, HarnessError, plain error, hostile throw, and timeoutMs all normalize."""
    ctx = await _setup()
    ctx.tools.register(
        _echo_tool(
            "coded",
            execute=lambda _a, _c: (_ for _ in ()).throw(
                HarnessError("disk full", "ENOSPC")
            ),
        )
    )
    ctx.tools.register(
        _echo_tool(
            "plain",
            execute=lambda _a, _c: (_ for _ in ()).throw(ValueError("bad input")),
        )
    )

    class _HostileErr(Exception):
        def __str__(self) -> str:
            raise RuntimeError("nope")

        def __repr__(self) -> str:
            raise RuntimeError("nope")

    ctx.tools.register(
        _echo_tool(
            "hostile",
            execute=lambda _a, _c: (_ for _ in ()).throw(_HostileErr()),
        )
    )

    async def _slow(_args: Any, run_ctx: Any) -> str:
        run_ctx["signal"].abort("cancelled mid-body")
        return "ignored"

    ctx.tools.register(_echo_tool("slow", execute=_slow))

    pipeline = ToolExecutionPipeline(ctx)
    sig = AbortSignal()

    unknown = await pipeline.run(
        {"callId": "c1", "name": "nope", "arguments": {}, "signal": sig}
    )
    assert unknown["isError"] is True
    assert unknown["error"] == {
        "message": 'unknown tool "nope"',
        "info": {"name": "ToolNotFoundError", "code": "UNKNOWN_TOOL"},
    }

    coded = await pipeline.run(
        {"callId": "c2", "name": "coded", "arguments": {}, "signal": sig}
    )
    assert coded["error"] == {
        "message": "disk full",
        "info": {"name": "HarnessError", "code": "ENOSPC"},
    }

    plain = await pipeline.run(
        {"callId": "c3", "name": "plain", "arguments": {}, "signal": sig}
    )
    assert plain["error"] == {"message": "bad input"}

    hostile = await pipeline.run(
        {"callId": "c4", "name": "hostile", "arguments": {}, "signal": sig}
    )
    assert hostile["content"] == [
        {"type": "text", "text": "Error: <unprintable thrown value>"}
    ]

    slow = await pipeline.run(
        {"callId": "c5", "name": "slow", "arguments": {}, "signal": sig}
    )
    assert slow["isError"] is True
    assert (
        "timeout" in slow["error"]["message"].lower()
        or slow["error"]["info"]["code"] == TOOL_ABORTED
    )


@pytest.mark.asyncio
async def test_output_validation_presentation_meta_and_post_execute() -> None:
    """Validates output schema, presentationMeta, value/content replacement, block, and finalizeContent."""
    ctx = await _setup()
    ctx.tools.register(
        define_tool(
            {
                "name": "projected",
                "description": "projected",
                "parameters": {"bad_out": {"type": "boolean"}},
                "output": {
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"text": {"type": "string", "required": True}},
                    },
                    "render": lambda _a, val: [
                        {"type": "text", "text": f"render:{val['text']}"}
                    ],
                    "presentationMeta": lambda _a, val: {"projected": val["text"]},
                },
                "execute": lambda a, _c: 123 if a.get("bad_out") else {"text": "body"},
            }
        )
    )

    pipeline = ToolExecutionPipeline(ctx)
    sig = AbortSignal()

    bad_out = await pipeline.run(
        {
            "callId": "b1",
            "name": "projected",
            "arguments": {"bad_out": True},
            "signal": sig,
        }
    )
    assert bad_out["isError"] is True
    assert bad_out["error"]["info"]["code"] == "INVALID_TOOL_OUTPUT"
    assert "value" not in bad_out

    mode = "content"
    seen_frozen: list[bool] = []

    async def _post(exec_info: Any, result: Any, next_fn: Any) -> Any:
        seen_frozen.append(is_frozen(result))
        if mode == "content":
            return {
                "kind": "accept",
                "content": [{"type": "text", "text": "policy content"}],
            }
        if mode == "value":
            return {
                "kind": "accept",
                "value": {"text": "policy value"},
                "additionalContexts": [
                    create_user_message(
                        content=[{"type": "text", "text": "extra ctx"}],
                        source={"kind": "test"},
                    )
                ],
            }
        if mode == "both":
            return {
                "kind": "accept",
                "value": {"text": "x"},
                "content": [{"type": "text", "text": "y"}],
            }
        if mode == "block":
            return {"kind": "block", "feedback": []}
        return await next_fn()

    unsub = ctx.on("tools/post-execute", _post, mode=EventMode.WATERFALL)

    res_content = await pipeline.run(
        {"callId": "p1", "name": "projected", "arguments": {}, "signal": sig}
    )
    assert seen_frozen == [True]
    assert res_content == {
        "isError": False,
        "value": {"text": "body"},
        "content": [{"type": "text", "text": "policy content"}],
        "meta": {"projected": "body"},
    }

    mode = "value"
    res_value = await pipeline.run(
        {"callId": "p2", "name": "projected", "arguments": {}, "signal": sig}
    )
    assert res_value["content"] == [{"type": "text", "text": "render:policy value"}]
    assert res_value["meta"] == {"projected": "policy value"}
    assert res_value["additionalContexts"][0]["content"] == [
        {"type": "text", "text": "extra ctx"}
    ]

    mode = "both"
    res_both = await pipeline.run(
        {"callId": "p3", "name": "projected", "arguments": {}, "signal": sig}
    )
    assert res_both["isError"] is True
    assert (
        res_both["error"]["message"]
        == "tools/post-execute accept decision cannot replace both value and content"
    )

    mode = "block"
    res_block = await pipeline.run(
        {"callId": "p4", "name": "projected", "arguments": {}, "signal": sig}
    )
    assert res_block["isError"] is True
    assert "value" not in res_block
    assert res_block["error"]["message"] == "tool result blocked by post-execute policy"
    unsub()

    # Snapshotted finalizeContent runs once even when tool disposed in pre-execute
    final_calls = 0

    def _finalize(_exec: Any, _res: Any) -> Any:
        nonlocal final_calls
        final_calls += 1
        return [{"type": "text", "text": "bounded failure"}]

    dispose = ctx.tools.register(_echo_tool("bounded", finalizeContent=_finalize))

    async def _pre_throw(_exec: Any, _next: Any) -> Any:
        dispose()
        raise HarnessError("policy failed", "POLICY_FAILED")

    ctx.on("tools/pre-execute", _pre_throw, mode=EventMode.WATERFALL)
    res_fin = await pipeline.run(
        {"callId": "f1", "name": "bounded", "arguments": {}, "signal": sig}
    )
    assert final_calls == 1
    assert res_fin["content"] == [{"type": "text", "text": "bounded failure"}]


@pytest.mark.asyncio
async def test_ask_routing_and_pre_execute_deny() -> None:
    """Tests pre-execute deny, degraded ask without approval/agent, and full approval routing."""
    ctx = await _setup()
    ctx.tools.register(_echo_tool())
    pipeline = ToolExecutionPipeline(ctx)
    sig = AbortSignal()

    unsub_ask = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "ask"},
        mode=EventMode.WATERFALL,
    )
    degraded = await pipeline.run(
        {"callId": "a1", "name": "echo", "arguments": {"text": "hi"}, "signal": sig}
    )
    assert degraded["isError"] is True
    assert (
        degraded["content"][0]["text"]
        == 'Error: tool "echo" requires approval (not yet supported)'
    )
    unsub_ask()

    class _FakeApproval(Service):
        def __init__(self, inner: Context) -> None:
            super().__init__(inner, "approval")
            self.outcome = "allowed-once"
            self.seen: list[dict[str, Any]] = []

        async def request(self, req: dict[str, Any]) -> str:
            self.seen.append(req)
            out = self.ctx.emit("approval/request", req)
            _ = out
            return self.outcome

    ctx.use(_FakeApproval)
    await ctx.flush()

    @dataclass
    class _FakeAgent:
        id: str

    agent = _FakeAgent("ag-1")
    ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "ask", "reason": "human check"},
        mode=EventMode.WATERFALL,
    )

    no_agent = await pipeline.run(
        {"callId": "a2", "name": "echo", "arguments": {"text": "hi"}, "signal": sig}
    )
    assert no_agent["isError"] is True
    assert "no agent to route it through" in no_agent["content"][0]["text"]

    allowed = await pipeline.run(
        {
            "callId": "a3",
            "name": "echo",
            "arguments": {"text": "hi"},
            "signal": sig,
            "agent": agent,
        }
    )
    assert allowed["isError"] is False
    assert allowed["value"] == "hi"

    ctx.approval.outcome = "rejected"
    rejected = await pipeline.run(
        {
            "callId": "a4",
            "name": "echo",
            "arguments": {},
            "signal": sig,
            "agent": agent,
        }
    )
    assert rejected["content"][0]["text"] == 'Error: the user rejected tool "echo"'

    ctx.approval.outcome = "cancelled"
    cancelled = await pipeline.run(
        {
            "callId": "a5",
            "name": "echo",
            "arguments": {},
            "signal": sig,
            "agent": agent,
        }
    )
    assert (
        cancelled["content"][0]["text"]
        == 'Error: approval for tool "echo" was cancelled'
    )

    ctx.approval.outcome = "unavailable"
    no_chan = await pipeline.run(
        {
            "callId": "a6",
            "name": "echo",
            "arguments": {},
            "signal": sig,
            "agent": agent,
        }
    )
    assert "no approval channel is available" in no_chan["content"][0]["text"]

    ctx.approval.outcome = "rogue-value"
    rogue = await pipeline.run(
        {
            "callId": "a7",
            "name": "echo",
            "arguments": {},
            "signal": sig,
            "agent": agent,
        }
    )
    assert rogue["isError"] is True
    assert "unreachable" in rogue["content"][0]["text"]

    # Abort overtaking pending approval returns ABORTED_BEFORE_DISPATCH
    gate = asyncio.Event()
    release: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def _slow_req(_req: dict[str, Any]) -> str:
        gate.set()
        return await release

    ctx.approval.request = _slow_req
    abort_sig = AbortSignal()
    pending = asyncio.create_task(
        pipeline.run(
            {
                "callId": "a8",
                "name": "echo",
                "arguments": {},
                "signal": abort_sig,
                "agent": agent,
            }
        )
    )
    await gate.wait()
    abort_sig.abort("caller stopped")
    release.set_result("allowed-once")
    overtaken = await pending
    assert overtaken["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH


@pytest.mark.asyncio
async def test_execute_tool_batch_concurrency_ordering_and_abort() -> None:
    """Parallel rolling pool, exclusive barrier, input-order notify, malformed JSON, scheduler error."""
    ctx = await _setup()
    timeline: list[str] = []
    started = 0
    both_running = asyncio.Event()

    async def _read(args: dict[str, Any], _ctx: Any) -> str:
        nonlocal started
        started += 1
        tag = args["tag"]
        timeline.append(f"start:{tag}")
        if started == 2:
            both_running.set()
        await asyncio.wait_for(both_running.wait(), timeout=2.0)
        timeline.append(f"end:{tag}")
        return tag

    async def _write(args: dict[str, Any], _ctx: Any) -> str:
        tag = args["tag"]
        timeline.append(f"exclusive:{tag}")
        return tag

    ctx.tools.register(
        define_tool(
            {
                "name": "r",
                "description": "read",
                "parameters": {"tag": {"type": "string", "required": True}},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "isConcurrencySafe": lambda _a: True,
                "execute": _read,
            }
        )
    )
    ctx.tools.register(
        define_tool(
            {
                "name": "w",
                "description": "write",
                "parameters": {"tag": {"type": "string", "required": True}},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "execute": _write,
            }
        )
    )
    notified: list[str] = []
    ctx.on("tools/result", lambda exec_info, _r: notified.append(exec_info["callId"]))

    pipeline = ToolExecutionPipeline(ctx)
    batch = await execute_tool_batch(
        {
            "signal": AbortSignal(),
            "calls": [
                {
                    "type": "tool-call",
                    "id": "c1",
                    "name": "r",
                    "arguments": '{"tag":"a"}',
                },
                {
                    "type": "tool-call",
                    "id": "c2",
                    "name": "r",
                    "arguments": '{"tag":"b"}',
                },
                {
                    "type": "tool-call",
                    "id": "c3",
                    "name": "w",
                    "arguments": '{"tag":"x"}',
                },
                {"type": "tool-call", "id": "c4", "name": "r", "arguments": "{bad"},
            ],
        },
        pipeline,
        max_parallel=2,
    )
    assert [r["callId"] for r in batch["results"]] == ["c1", "c2", "c3", "c4"]
    assert batch["results"][3]["result"]["isError"] is True
    assert timeline.index("exclusive:x") > timeline.index("end:a")
    assert timeline.index("exclusive:x") > timeline.index("end:b")
    assert notified == ["c1", "c2", "c3", "c4"]

    # Scheduler exception propagates via ToolBatchExecutionError preserving settled prefix
    def _exploding_max_parallel() -> int:
        raise RuntimeError("scheduler blew up")

    with pytest.raises(ToolBatchExecutionError):
        await execute_tool_batch(
            {
                "signal": AbortSignal(),
                "calls": [
                    {
                        "type": "tool-call",
                        "id": "s1",
                        "name": "w",
                        "arguments": '{"tag":"1"}',
                    }
                ],
            },
            pipeline,
            max_parallel=_exploding_max_parallel,
        )

    # Scoped notify_result routes to matching scope chain only
    agent_a = {"id": "a"}
    agent_b = {"id": "b"}
    scope_a = create_scope(ctx, agent_a)
    scope_b = create_scope(ctx, agent_b)
    seen_scopes: list[str] = []
    scope_a.ctx.on("tools/result", lambda exec_info, _r: seen_scopes.append("a"))
    scope_b.ctx.on("tools/result", lambda exec_info, _r: seen_scopes.append("b"))
    await pipeline.run(
        {
            "callId": "sc1",
            "name": "w",
            "arguments": {"tag": "s"},
            "signal": AbortSignal(),
            "agent": agent_a,
        }
    )
    assert seen_scopes == ["a"]


@pytest.mark.asyncio
async def test_pipeline_additional_edge_paths_and_batch_backpressure() -> None:
    """Covers remaining timeout, abort, post-execute, parser, and batch pool branches."""
    ctx = await _setup()
    ctx.tools.register(
        _echo_tool(
            "async_ok",
            execute=lambda args, _c: asyncio.sleep(0, result=args.get("text", "ok")),
        )
    )
    ctx.tools.register(_echo_tool("sync_ok"))

    async def _hung(_args: Any, run_ctx: Any) -> str:
        run_ctx["signal"].abort("cancelled mid-run")
        await asyncio.sleep(0)
        return "late"

    ctx.tools.register(_echo_tool("hung", execute=_hung))

    async def _coop_abort(_args: Any, run_ctx: Any) -> str:
        run_ctx["signal"].abort("mid-body")
        return "ignored"

    # Body abort via caller signal mid-execution & exception after caller signal aborts
    ext_sig = AbortSignal()

    async def _sig_return(_args: Any, _c: Any) -> str:
        ext_sig.abort("caller cancelled")
        return "ignored"

    async def _sig_raise(_args: Any, _c: Any) -> str:
        ext_sig.abort("caller raised")
        raise RuntimeError("boom after abort")

    ctx.tools.register(_echo_tool("sig_ret", execute=_sig_return))
    ctx.tools.register(_echo_tool("sig_err", execute=_sig_raise))
    ctx.tools.register(
        _echo_tool(
            "with_fin_none",
            finalizeContent=lambda _e, _r: None,
        )
    )

    pipeline = ToolExecutionPipeline(ctx)

    # Async without timeout & hung timeout via asyncio.wait_for TimeoutError
    ok_res = await pipeline.run(
        {
            "callId": "p1",
            "name": "async_ok",
            "arguments": {"text": "hi"},
            "signal": AbortSignal(),
        }
    )
    assert ok_res["value"] == "hi"
    hung_res = await pipeline.run(
        {"callId": "p2", "name": "hung", "arguments": {}, "signal": AbortSignal()}
    )
    assert hung_res["error"]["info"]["code"] == TOOL_ABORTED

    # Caller abort mid-body (both return and raise paths)
    ret_aborted = await pipeline.run(
        {"callId": "p3", "name": "sig_ret", "arguments": {}, "signal": ext_sig}
    )
    assert ret_aborted["error"]["info"]["code"] == TOOL_ABORTED
    ext_sig2 = AbortSignal()

    async def _sig_raise2(_args: Any, _c: Any) -> str:
        ext_sig2.abort("caller raised 2")
        raise RuntimeError("boom")

    ctx.tools.register(_echo_tool("sig_err2", execute=_sig_raise2))
    err_aborted = await pipeline.run(
        {"callId": "p4", "name": "sig_err2", "arguments": {}, "signal": ext_sig2}
    )
    assert err_aborted["error"]["message"] == "boom"

    # Pre-execute returning None, overriding args, aborting inside pre-execute, and throwing
    ctx.on("tools/pre-execute", lambda _e, _n: None, mode=EventMode.WATERFALL)
    r_none = await pipeline.run(
        {
            "callId": "p5",
            "name": "sync_ok",
            "arguments": {"text": "kept"},
            "signal": AbortSignal(),
        }
    )
    assert r_none["value"] == "kept"

    pre_sig = AbortSignal()

    async def _pre_abort(_e: Any, _n: Any) -> Any:
        pre_sig.abort("aborted in pre")
        return {"kind": "allow"}

    dispose_pre_abort = ctx.on(
        "tools/pre-execute", _pre_abort, prepend=True, mode=EventMode.WATERFALL
    )
    r_pre_ab = await pipeline.run(
        {"callId": "p6", "name": "sync_ok", "arguments": {}, "signal": pre_sig}
    )
    assert r_pre_ab["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    dispose_pre_abort()

    dispose_pre_err = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: (_ for _ in ()).throw(RuntimeError("pre fail")),
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    r_pre_err = await pipeline.run(
        {"callId": "p7", "name": "sync_ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert r_pre_err["error"]["message"] == "pre fail"
    dispose_pre_err()

    dispose_pre_throw_ab = ctx.on(
        "tools/pre-execute",
        lambda e, _n: (
            e["signal"].abort("ab"),
            (_ for _ in ()).throw(RuntimeError("x")),
        ),
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    r_pre_th_ab = await pipeline.run(
        {"callId": "p8", "name": "sync_ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert r_pre_th_ab["error"]["message"] == "x"
    dispose_pre_throw_ab()

    # Pre-execute allow extra keys never rewrite arguments; cancel returns ABORTED_BEFORE_DISPATCH
    dispose_arg_ovr = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "allow", "arguments": {"text": "overridden"}},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    r_ovr = await pipeline.run(
        {
            "callId": "p9",
            "name": "with_fin_none",
            "arguments": {"text": "orig"},
            "signal": AbortSignal(),
        }
    )
    assert r_ovr["value"] == "orig"
    dispose_arg_ovr()

    dispose_cancel = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "cancel"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    r_cancel = await pipeline.run(
        {
            "callId": "p9c",
            "name": "with_fin_none",
            "arguments": {"text": "orig"},
            "signal": AbortSignal(),
        }
    )
    assert r_cancel["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    dispose_cancel()

    # execute waterfall pre-abort, non-abort return before body, invalid result, rich error, rich success
    exec_ab_sig = AbortSignal()
    dispose_ex_ab = ctx.on(
        "tools/execute",
        lambda _e, _n: (
            exec_ab_sig.abort("stop"),
            {"isError": False, "value": "x", "content": []},
        )[1],
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    r_ex_ab = await pipeline.run(
        {"callId": "p10", "name": "sync_ok", "arguments": {}, "signal": exec_ab_sig}
    )
    assert r_ex_ab["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    dispose_ex_ab()

    dispose_ex_inv = ctx.on(
        "tools/execute", lambda _e, _n: "bad", prepend=True, mode=EventMode.WATERFALL
    )
    bad_ex_res = await pipeline.run(
        {
            "callId": "p11",
            "name": "sync_ok",
            "arguments": {},
            "signal": AbortSignal(),
        }
    )
    assert bad_ex_res["isError"] is True
    dispose_ex_inv()

    extra_msg = create_user_message(content=[{"type": "text", "text": "ctx msg"}])
    dispose_ex_err = ctx.on(
        "tools/execute",
        lambda _e, _n: {
            "isError": True,
            "error": {"message": "rich", "info": {"k": 1}},
            "content": [{"type": "text", "text": "Error: rich"}],
            "meta": {"m": 1},
            "additionalContexts": [extra_msg],
        },
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    r_rich_err = await pipeline.run(
        {"callId": "p12", "name": "sync_ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert r_rich_err["meta"] == {"m": 1}
    assert r_rich_err["additionalContexts"] == [extra_msg]
    dispose_ex_err()

    dispose_ex_ok = ctx.on(
        "tools/execute",
        lambda _e, _n: {
            "isError": False,
            "value": "val",
            "content": [{"type": "text", "text": "override"}],
            "meta": {"custom": 2},
            "additionalContexts": [extra_msg],
        },
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    r_rich_ok = await pipeline.run(
        {"callId": "p13", "name": "sync_ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert r_rich_ok["content"] == [{"type": "text", "text": "val"}]
    assert "meta" not in r_rich_ok
    assert r_rich_ok["additionalContexts"] == [extra_msg]
    dispose_ex_ok()

    # Live tool disappearing before _normalize_dispatch_result completes
    dispose_vanish = ctx.tools.register(_echo_tool("vanish"))

    async def _vanish_exec(ex: Any, next_fn: Any) -> Any:
        dispose_vanish()
        return {"isError": False, "value": "v", "content": []}

    dispose_ex_vanish = ctx.on(
        "tools/execute", _vanish_exec, prepend=True, mode=EventMode.WATERFALL
    )
    r_vanish = await pipeline.run(
        {"callId": "p14", "name": "vanish", "arguments": {}, "signal": AbortSignal()}
    )
    assert r_vanish["error"]["info"]["code"] == "UNKNOWN_TOOL"
    dispose_ex_vanish()

    # Post-execute returning None, throwing, block with additionalContexts, non-list additionalContexts
    dispose_post_none = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: None,
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        await pipeline.run(
            {
                "callId": "p15",
                "name": "sync_ok",
                "arguments": {"text": "ok"},
                "signal": AbortSignal(),
            }
        )
    )["value"] == "ok"
    dispose_post_none()

    dispose_post_throw = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: (_ for _ in ()).throw(RuntimeError("post threw")),
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        await pipeline.run(
            {
                "callId": "p16",
                "name": "sync_ok",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["error"]["message"] == "post threw"
    dispose_post_throw()

    dispose_post_block = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {
            "kind": "block",
            "feedback": [{"type": "text", "text": "blocked"}],
            "additionalContexts": [extra_msg],
        },
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    r_blk = await pipeline.run(
        {"callId": "p17", "name": "sync_ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert r_blk["additionalContexts"] == [extra_msg]
    dispose_post_block()

    dispose_post_bad_ctx = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {"kind": "accept", "additionalContexts": "not-list"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        "must be a list"
        in (
            await pipeline.run(
                {
                    "callId": "p18",
                    "name": "sync_ok",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    dispose_post_bad_ctx()

    # Non-list content in render()
    ctx.tools.register(
        define_tool(
            {
                "name": "bad_render",
                "description": "bad render",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, _v: "not-a-list",
                },
                "execute": lambda _a, _c: "x",
            }
        )
    )
    assert (
        "non-lossless JSON"
        in (
            await pipeline.run(
                {
                    "callId": "p19",
                    "name": "bad_render",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    dispose_post_bad_ctx_noop = None
    _ = dispose_post_bad_ctx_noop

    # notify_result exception containment
    with patch(
        "magitech_harness.tools.pipeline.emit_scoped",
        side_effect=RuntimeError("notify fail"),
    ):
        await pipeline.run(
            {
                "callId": "p20",
                "name": "sync_ok",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )

    # Batch argument validation variants, deferContext/concludeTurn via ToolRunContext, max_parallel=1 pool,
    # and mid-batch failure preserving partial settled calls
    async def _par_async(_a: Any, run_ctx: Any) -> str:
        run_ctx["deferContext"](extra_msg)
        run_ctx["concludeTurn"]()
        await asyncio.sleep(0)
        return "done"

    ctx.tools.register(
        define_tool(
            {
                "name": "par",
                "description": "parallel",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "isConcurrencySafe": lambda _a: True,
                "execute": _par_async,
            }
        )
    )
    batch_res = await execute_tool_batch(
        {
            "signal": AbortSignal(),
            "calls": [
                {"type": "tool-call", "id": "b1", "name": "par", "arguments": ""},
                {"type": "tool-call", "id": "b1b", "name": "par", "arguments": "{}"},
                {"type": "tool-call", "id": "b2", "name": "par", "arguments": 123},
                {"type": "tool-call", "id": "b3", "name": "par", "arguments": "[1]"},
                {"type": "tool-call", "id": "b4", "name": "par", "arguments": {"x": 1}},
            ],
        },
        pipeline,
        max_parallel=1,
    )
    assert batch_res["results"][0]["result"]["value"] == "done"
    assert batch_res["results"][1]["result"]["value"] == "done"
    assert batch_res["results"][2]["result"]["isError"] is True
    assert batch_res["results"][3]["result"]["isError"] is True
    assert batch_res["results"][4]["result"]["value"] == "done"
    assert batch_res["concluded"] is True
    assert len(batch_res["additionalContexts"]) == 3

    # Batch scheduler failing after 1 parallel task is in flight preserves partial settled prefix
    calls_mp = 0

    def _flaky_max_parallel() -> int:
        nonlocal calls_mp
        calls_mp += 1
        if calls_mp == 2:
            raise RuntimeError("boom on second")
        return 2

    with pytest.raises(ToolBatchExecutionError) as batch_exc:
        await execute_tool_batch(
            {
                "signal": AbortSignal(),
                "calls": [
                    {
                        "type": "tool-call",
                        "id": "f1",
                        "name": "sync_ok",
                        "arguments": "{}",
                    },
                    {"type": "tool-call", "id": "f2", "name": "par", "arguments": "{}"},
                ],
            },
            pipeline,
            max_parallel=_flaky_max_parallel,
        )
    assert len(batch_exc.value.partial["results"]) == 1
    assert batch_exc.value.partial["results"][0]["callId"] == "f1"


@pytest.mark.asyncio
async def test_pipeline_remaining_branch_matrix() -> None:
    """Covers remaining helper, ask, dispatch, normalization, and post-execute branches."""

    # Hostile mapping & empty-message exception in tool_error_message + AbortError + _snapshot_lossless(None)
    class _BadMapping(dict[str, Any]):
        def get(self, key: Any, default: Any = None) -> Any:
            raise RuntimeError("bad get")

    assert tool_error_message(_BadMapping()) == "<unprintable thrown value>"
    assert tool_error_message({"code": 1}) == "{'code': 1}"
    assert tool_error_message(ValueError()) == "ValueError"

    ctx = await _setup()
    ctx.tools.register(
        _echo_tool(
            "throws_abort",
            execute=lambda _a, _c: (_ for _ in ()).throw(AbortError("stopped")),
        )
    )
    ctx.tools.register(
        _echo_tool(
            "nonlossless_val",
            execute=lambda _a, _c: float("inf"),
        )
    )
    ctx.tools.register(
        _echo_tool(
            "async_fast_ok",
            execute=lambda _a, _c: asyncio.sleep(0, result="quick"),
        )
    )
    ctx.tools.register(_echo_tool("ok"))
    ctx.tools.register(
        _echo_tool(
            "fin_boom",
            finalizeContent=lambda _e, _r: (_ for _ in ()).throw(
                RuntimeError("fin boom")
            ),
        )
    )
    pipeline = ToolExecutionPipeline(ctx)

    abort_err_res = await pipeline.run(
        {
            "callId": "a1",
            "name": "throws_abort",
            "arguments": {},
            "signal": AbortSignal(),
        }
    )
    assert abort_err_res["isError"] is True
    assert abort_err_res["error"] == {"message": "stopped"}
    assert (
        await pipeline.run(
            {
                "callId": "a2",
                "name": "nonlossless_val",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["error"]["info"]["code"] == "INVALID_TOOL_OUTPUT"
    assert (
        await pipeline.run(
            {
                "callId": "a3",
                "name": "async_fast_ok",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["value"] == "quick"
    assert (
        await pipeline.run(
            {
                "callId": "a4",
                "name": "fin_boom",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["error"]["message"] == "fin boom"

    # Direct prepare() + complete() when signal is already aborted + prepare with parent
    pre_ab = AbortSignal()
    pre_ab.abort("prior")
    prep_ab = await pipeline.prepare(
        {
            "callId": "d1",
            "name": "ok",
            "arguments": {},
            "signal": pre_ab,
            "parent": object(),
        }
    )
    assert prep_ab["kind"] == "final-result"
    assert prep_ab["result"]["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    completed_ab = await pipeline.complete(prep_ab["exec"], prep_ab)
    assert completed_ab["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH

    # Approval displayReason + reason + rejected, and ask without reason
    @dataclass
    class _RejApproval:
        last_req: dict[str, Any] | None = None

        async def request(self, req: dict[str, Any]) -> str:
            self.last_req = req
            return "rejected"

    app = _RejApproval()
    ctx.provide("approval", app)
    off_ask = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {
            "kind": "ask",
            "reason": "danger",
            "displayReason": "ui danger",
        },
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    rej = await pipeline.run(
        {
            "callId": "k1",
            "name": "ok",
            "arguments": {},
            "signal": AbortSignal(),
            "agent": {"id": "ag"},
        }
    )
    assert "rejected" in rej["error"]["message"]
    assert app.last_req is not None and app.last_req["displayReason"] == "ui danger"
    off_ask()

    off_ask_bare = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "ask"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    rej_bare = await pipeline.run(
        {
            "callId": "k1b",
            "name": "ok",
            "arguments": {},
            "signal": AbortSignal(),
            "agent": {"id": "ag"},
        }
    )
    assert "rejected" in rej_bare["error"]["message"]
    assert app.last_req is not None and "reason" not in app.last_req
    off_ask_bare()

    # presentationMeta None + non-lossless presentationMeta error
    ctx.tools.register(
        define_tool(
            {
                "name": "meta_none",
                "description": "meta none",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                    "presentationMeta": lambda _a, _v: None,
                },
                "execute": lambda _a, _c: "ok",
            }
        )
    )
    ctx.tools.register(
        define_tool(
            {
                "name": "meta_bad",
                "description": "meta bad",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                    "presentationMeta": lambda _a, _v: float("inf"),
                },
                "execute": lambda _a, _c: "ok",
            }
        )
    )
    assert (
        await pipeline.run(
            {
                "callId": "m1",
                "name": "meta_none",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["meta"] is None
    assert (
        "non-lossless JSON"
        in (
            await pipeline.run(
                {
                    "callId": "m2",
                    "name": "meta_bad",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )

    # pre-execute deny with info
    off_deny = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "deny", "reason": "denied", "info": {"policy": 1}},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    den_res = await pipeline.run(
        {"callId": "k2", "name": "ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert den_res["error"]["info"] == {"policy": 1}
    off_deny()

    # tools/execute middleware aborts signal before next_fn, plus throws before/after body
    async def _abort_before_next(ex: Any, next_fn: Any) -> Any:
        ex["signal"].abort("aborted before default")
        return await next_fn(ex)

    off_ab_next = ctx.on(
        "tools/execute", _abort_before_next, prepend=True, mode=EventMode.WATERFALL
    )
    assert (
        await pipeline.run(
            {"callId": "k3", "name": "ok", "arguments": {}, "signal": AbortSignal()}
        )
    )["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    off_ab_next()

    off_ex_throw_pre = ctx.on(
        "tools/execute",
        lambda e, _n: (
            e["signal"].abort("ab pre throw"),
            (_ for _ in ()).throw(RuntimeError("x")),
        ),
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        await pipeline.run(
            {"callId": "k4", "name": "ok", "arguments": {}, "signal": AbortSignal()}
        )
    )["error"]["message"] == "x"
    off_ex_throw_pre()

    off_ex_throw = ctx.on(
        "tools/execute",
        lambda _e, _n: (_ for _ in ()).throw(RuntimeError("exec threw")),
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        await pipeline.run(
            {"callId": "k5", "name": "ok", "arguments": {}, "signal": AbortSignal()}
        )
    )["error"]["message"] == "exec threw"
    off_ex_throw()

    # tools/execute wrapper throwing after body completes preserves thrown error; aborting without throw returns TOOL_ABORTED
    ctx.tools.register(
        _echo_tool(
            "raise_after_abort",
            execute=lambda _a, _c: "ok",
        )
    )

    async def _wrap_body_abort_throw(ex: Any, next_fn: Any) -> Any:
        await next_fn(ex)
        ex["signal"].abort("post body")
        raise RuntimeError("outer throw")

    off_post_abort = ctx.on(
        "tools/execute", _wrap_body_abort_throw, prepend=True, mode=EventMode.WATERFALL
    )
    assert (
        await pipeline.run(
            {
                "callId": "k6",
                "name": "raise_after_abort",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["error"]["message"] == "outer throw"
    off_post_abort()

    # Error without info, None meta, and non-canonical success re-rendering through output contract
    off_err_noinfo = ctx.on(
        "tools/execute",
        lambda _e, _n: {
            "isError": True,
            "error": {"message": "plain err"},
            "content": [{"type": "text", "text": "Error: plain err"}],
            "meta": None,
        },
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    err_noinfo = await pipeline.run(
        {"callId": "k7", "name": "ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert err_noinfo["error"] == {"message": "plain err"}
    off_err_noinfo()

    off_ok_empty = ctx.on(
        "tools/execute",
        lambda _e, _n: {"isError": False, "value": "rendered", "content": []},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    ok_empty = await pipeline.run(
        {"callId": "k8", "name": "ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert ok_empty["content"] == [{"type": "text", "text": "rendered"}]
    off_ok_empty()

    # post-execute replacing value on error, replacing value on vanished tool, preserving dispatched contexts,
    # and replacing content + additionalContexts
    extra_c = create_user_message(content=[{"type": "text", "text": "more"}])
    off_post_val_on_err = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {"kind": "accept", "value": "x"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        "failed result"
        in (
            await pipeline.run(
                {
                    "callId": "k9",
                    "name": "nonlossless_val",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    off_post_val_on_err()

    dispose_temp = ctx.tools.register(_echo_tool("vanish_post"))

    async def _vanish_in_post(_ex: Any, _res: Any, _n: Any) -> Any:
        dispose_temp()
        return {"kind": "accept", "value": "new"}

    off_post_vanish = ctx.on(
        "tools/post-execute", _vanish_in_post, prepend=True, mode=EventMode.WATERFALL
    )
    assert (
        await pipeline.run(
            {
                "callId": "k10",
                "name": "vanish_post",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["error"]["info"]["code"] == "UNKNOWN_TOOL"
    off_post_vanish()

    off_ex_with_ctx = ctx.on(
        "tools/execute",
        lambda _e, _n: {
            "isError": False,
            "value": "orig",
            "content": [],
            "additionalContexts": [extra_c],
        },
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    off_post_keep_ctx = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {"kind": "accept", "value": "replaced"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    kept = await pipeline.run(
        {"callId": "k11", "name": "ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert kept["value"] == "replaced"
    assert kept["additionalContexts"] == [extra_c]
    off_post_keep_ctx()
    off_ex_with_ctx()

    off_post_content_ctx = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {
            "kind": "accept",
            "content": [{"type": "text", "text": "c"}],
            "additionalContexts": [extra_c],
        },
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    replaced_c = await pipeline.run(
        {"callId": "k12", "name": "ok", "arguments": {}, "signal": AbortSignal()}
    )
    assert replaced_c["content"] == [{"type": "text", "text": "c"}]
    assert replaced_c["additionalContexts"] == [extra_c]
    off_post_content_ctx()

    off_post_val_plain = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {"kind": "accept", "value": "plain_val"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        await pipeline.run(
            {"callId": "k13", "name": "ok", "arguments": {}, "signal": AbortSignal()}
        )
    )["value"] == "plain_val"
    off_post_val_plain()


@pytest.mark.asyncio
async def test_pipeline_rc1_parity_and_batch_logging() -> None:
    """Covers ToolExecutionSource, projectContent, signal fusion, validation, and BatchSettlement Session logging."""
    ctx = await _setup()

    # render_content_summary, tool_error_message attribute, and hostile HarnessError.__getattribute__
    assert (
        render_content_summary(
            [{"type": "text", "text": "a"}, {"type": "image", "data": "b"}]
        )
        == "a\n[image content]"
    )
    assert render_content_summary([{"type": "text", "text": ""}]) == (
        "tool result blocked by post-execute policy"
    )
    assert tool_error_message(SimpleNamespace(message="attr msg")) == "attr msg"

    class _HostileCode(HarnessError):
        @property
        def code(self) -> str:  # type: ignore[override]
            raise RuntimeError("boom code")

        @code.setter
        def code(self, _value: str) -> None:
            pass

    assert _HostileCode("bad code", "BAD").message == "bad code"

    # fuse_tool_signals when already aborted & wrapper signal abort before dispatch
    pre_sig = AbortSignal()
    pre_sig.abort("already")
    fused_pre, disp_pre = fuse_tool_signals(pre_sig, AbortSignal())
    assert fused_pre.is_aborted is True
    disp_pre()

    # ToolExecutionSource: get & reject, pre-aborted refusal precedence, projectContent skip on refuse
    events_log: list[str] = []
    projected_tool = define_tool(
        {
            "name": "proj",
            "description": "projected tool",
            "parameters": {"x": {"type": "string"}},
            "output": {
                "schema": {"type": "string"},
                "render": lambda _a, v: [{"type": "text", "text": f"raw:{v}"}],
            },
            "projectContent": lambda _e, res: (
                events_log.append("project"),
                [{"type": "text", "text": f"proj:{res['content'][0]['text']}"}],
            )[1],
            "finalizeContent": lambda _e, res: (
                events_log.append("finalize"),
                [{"type": "text", "text": f"fin:{res['content'][0]['text']}"}],
            )[1],
            "execute": lambda a, _c: a.get("x", "v"),
        }
    )

    class _Source:
        def __init__(self, refuse: bool = True) -> None:
            self._refuse = refuse

        def get(self, name: str, _scope: Any = None, _ts: Any = None) -> Any:
            return projected_tool if name == "proj" else None

        def reject(self, _name: str, nested: bool) -> Any:
            if self._refuse and nested:
                return ValueError("refused direct nested call")
            return None

    src_pipeline = ToolExecutionPipeline(ctx, source=_Source(refuse=True))
    # Refused without abort -> error returned, finalizeContent skipped
    refused = await src_pipeline.run(
        {
            "callId": "src1",
            "name": "proj",
            "arguments": {},
            "parent": object(),
            "signal": AbortSignal(),
        }
    )
    assert refused["isError"] is True
    assert refused["error"]["message"] == "refused direct nested call"
    assert events_log == []

    # Refused WITH pre-aborted signal -> ABORTED_BEFORE_DISPATCH takes precedence, finalizeContent runs
    src_pre_abort = AbortSignal()
    src_pre_abort.abort("pre")
    refused_aborted = await src_pipeline.run(
        {
            "callId": "src2",
            "name": "proj",
            "arguments": {},
            "parent": object(),
            "signal": src_pre_abort,
        }
    )
    assert refused_aborted["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    assert events_log == ["finalize"]

    # Source allow (reject returns None) -> projectContent runs before post-execute, then finalizeContent
    events_log.clear()
    observed_exec: list[Any] = []
    ctx.on("tools/result", lambda ex, _r: observed_exec.append(ex))
    allowed_src = ToolExecutionPipeline(ctx, source=_Source(refuse=False))
    post_seen: list[str] = []

    async def _see_proj(ex: Any, res: Any, next_fn: Any) -> Any:
        post_seen.append(res["content"][0]["text"])
        assert ex["rootCallId"] == "root-1"
        assert ex["schema"] == {"name": "proj"}
        return await next_fn()

    off_see_proj = ctx.on("tools/post-execute", _see_proj, mode=EventMode.WATERFALL)
    proj_ok = await allowed_src.run(
        {
            "callId": "src3",
            "rootCallId": "root-1",
            "schema": {"name": "proj"},
            "name": "proj",
            "arguments": {"x": "hi"},
            "signal": AbortSignal(),
        }
    )
    off_see_proj()
    assert post_seen == ["proj:raw:hi"]
    assert proj_ok["content"] == [{"type": "text", "text": "fin:proj:raw:hi"}]
    assert isinstance(observed_exec[-1], FrozenDict)

    # projectContent returning None leaves content untouched; projectContent throwing errors cleanly
    proj_none = define_tool(
        {
            "name": "proj_none",
            "description": "d",
            "parameters": {},
            "output": {
                "schema": {"type": "string"},
                "render": lambda _a, v: [{"type": "text", "text": v}],
            },
            "projectContent": lambda _e, _r: None,
            "execute": lambda _a, _c: "ok",
        }
    )
    proj_boom = define_tool(
        {
            "name": "proj_boom",
            "description": "d",
            "parameters": {},
            "output": {
                "schema": {"type": "string"},
                "render": lambda _a, v: [{"type": "text", "text": v}],
            },
            "projectContent": lambda _e, _r: (_ for _ in ()).throw(
                RuntimeError("proj boom")
            ),
            "execute": lambda _a, _c: "ok",
        }
    )
    ctx.tools.register(proj_none)
    ctx.tools.register(proj_boom)
    assert (
        await ToolExecutionPipeline(ctx).run(
            {
                "callId": "pn1",
                "name": "proj_none",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["content"] == [{"type": "text", "text": "ok"}]
    assert (
        await ToolExecutionPipeline(ctx).run(
            {
                "callId": "pb1",
                "name": "proj_boom",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["error"]["message"] == "proj boom"

    # Non-lossless arguments (even pre-aborted) & missing arguments key
    pipeline = ToolExecutionPipeline(ctx)
    bad_args_ab = AbortSignal()
    bad_args_ab.abort("stop")
    bad_args_res = await pipeline.run(
        {
            "callId": "badarg1",
            "name": "proj_none",
            "arguments": {"fn": lambda: None},
            "signal": bad_args_ab,
        }
    )
    assert (
        bad_args_res["error"]["message"]
        == "tool execution arguments must be losslessly JSON-serializable"
    )
    missing_args_res = await pipeline.run(
        {"callId": "badarg2", "name": "proj_none", "signal": AbortSignal()}  # type: ignore[typeddict-item]
    )
    assert (
        missing_args_res["error"]["message"]
        == "tool execution arguments must be losslessly JSON-serializable"
    )

    # Hostile HarnessError.__code__ in _error_info + hostile object without .code inside finish
    hostile_code_tool = define_tool(
        {
            "name": "hostile_code",
            "description": "d",
            "parameters": {},
            "output": {"schema": {"type": "null"}, "render": lambda _a, _v: []},
            "execute": lambda _a, _c: (_ for _ in ()).throw(
                _HostileCode("bad code", "BAD")
            ),
        }
    )
    ctx.tools.register(hostile_code_tool)
    hc_res = await pipeline.run(
        {
            "callId": "hc1",
            "name": "hostile_code",
            "arguments": {},
            "signal": AbortSignal(),
        }
    )
    assert hc_res["error"] == {"message": "bad code"}

    # Missing state invariant checks + complete() invalid kind
    bad_dispatch = await pipeline.dispatch(
        {"callId": "x", "name": "proj_none", "arguments": {}, "signal": AbortSignal()}
    )
    assert "missing execution state" in bad_dispatch["result"]["error"]["message"]
    bad_complete = await pipeline.complete(
        {"callId": "x", "name": "proj_none", "token": object()},  # type: ignore[dict-item]
        {"kind": "post-result", "result": hc_res},
    )
    assert "missing execution state" in bad_complete["error"]["message"]
    with pytest.raises(RuntimeError, match="missing execution state"):
        pipeline._require_state({"token": object()})
    with pytest.raises(ValueError, match="scheduled tool outcome"):
        await pipeline.complete({"callId": "x"}, {"kind": "invalid"})

    # prepare(): invalid pre-execute decision, empty deny reason, abort after check_guards, sync approval
    off_bad_pre = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "weird"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        "invalid tools/pre-execute decision"
        in (
            await pipeline.run(
                {
                    "callId": "pr1",
                    "name": "proj_none",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    off_bad_pre()

    off_empty_deny = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "deny", "reason": ""},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        "non-empty reason"
        in (
            await pipeline.run(
                {
                    "callId": "pr2",
                    "name": "proj_none",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    off_empty_deny()

    post_guard_sig = AbortSignal()
    off_guard_abort = ctx.tools.guard(
        lambda _e: (post_guard_sig.abort("after guard"), None)[1]
    )
    assert (
        await pipeline.run(
            {
                "callId": "pr3",
                "name": "proj_none",
                "arguments": {},
                "signal": post_guard_sig,
            }
        )
    )["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    off_guard_abort()

    # Sync approval service outcome
    ctx.provide("approval", SimpleNamespace(request=lambda _req: "allowed-once"))
    off_sync_ask = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "ask"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        await pipeline.run(
            {
                "callId": "pr4",
                "name": "proj_none",
                "arguments": {},
                "agent": {"id": "sync_ag"},
                "signal": AbortSignal(),
            }
        )
    )["value"] == "ok"
    off_sync_ask()

    # dispatch(): wrapper replaced signal already aborted before _dispatch_tool_body
    off_wrap_pre_ab = ctx.on(
        "tools/execute",
        lambda ex, next_fn: (
            ex.__setitem__("signal", fused_pre),
            next_fn(ex),
        )[1],
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        await pipeline.run(
            {
                "callId": "dp1",
                "name": "proj_none",
                "arguments": {},
                "signal": AbortSignal(),
            }
        )
    )["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    off_wrap_pre_ab()

    # dispatch(): non-canonical success missing "value", and deferContext + wrapper additionalContexts merge
    off_no_val = ctx.on(
        "tools/execute",
        lambda _ex, _n: {"isError": False, "content": []},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        "value is not lossless JSON"
        in (
            await pipeline.run(
                {
                    "callId": "dp2",
                    "name": "proj_none",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    off_no_val()

    ctx_msg1 = create_user_message(content=[{"type": "text", "text": "deferred"}])
    ctx_msg2 = create_user_message(content=[{"type": "text", "text": "wrapped"}])
    ctx.tools.register(
        define_tool(
            {
                "name": "defer_tool",
                "description": "d",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "execute": lambda _a, r: (r["deferContext"](ctx_msg1), "done")[1],
            }
        )
    )
    off_wrap_ctx = ctx.on(
        "tools/execute",
        lambda ex, next_fn: _wrap_add_ctx(ex, next_fn, ctx_msg2),
        prepend=True,
        mode=EventMode.WATERFALL,
    )

    async def _wrap_add_ctx(ex: Any, next_fn: Any, extra: Any) -> Any:
        res = await next_fn(ex)
        return {**res, "additionalContexts": [extra]}

    merged_ctx_res = await pipeline.run(
        {
            "callId": "dp3",
            "name": "defer_tool",
            "arguments": {},
            "signal": AbortSignal(),
        }
    )
    assert merged_ctx_res["additionalContexts"] == [ctx_msg1, ctx_msg2]
    off_wrap_ctx()

    # _post_execute: invalid kind & non-list block feedback & finalizeContent returning non-lossless content
    off_bad_post = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {"kind": "invalid-post"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        "invalid tools/post-execute decision"
        in (
            await pipeline.run(
                {
                    "callId": "pe1",
                    "name": "proj_none",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    off_bad_post()

    off_bad_blk = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {"kind": "block", "feedback": "bad"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        "must be a list"
        in (
            await pipeline.run(
                {
                    "callId": "pe2",
                    "name": "proj_none",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    off_bad_blk()

    # _finish non-lossless result from post-execute fallback + non-lossless finalizeContent
    ctx.tools.register(
        _echo_tool(
            "fin_nan",
            finalizeContent=lambda _e, _r: [{"type": "text", "text": float("nan")}],
        )  # type: ignore[dict-item]
    )
    assert (
        "losslessly JSON-serializable"
        in (
            await pipeline.run(
                {
                    "callId": "pe3",
                    "name": "fin_nan",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    off_post_nan = ctx.on(
        "tools/post-execute",
        lambda _e, _r, _n: {
            "kind": "accept",
            "content": [{"type": "text", "text": float("inf")}],
        },
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    assert (
        "losslessly JSON-serializable"
        in (
            await pipeline.run(
                {
                    "callId": "pe4",
                    "name": "proj_none",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )
    off_post_nan()

    # parse_arguments & execute_tool_batch validation & Session BatchSettlement logging & mid-batch abort & dispatch failure drain
    assert parse_arguments({"k": 1}) == {"k": 1}
    assert parse_arguments("") == {}
    assert parse_arguments("{bad") == "{bad"

    for bad_limit in (0, -1, True, "2"):
        with pytest.raises(ValueError, match="positive integer"):
            await execute_tool_batch(
                {
                    "signal": AbortSignal(),
                    "calls": [],
                    "maxParallelToolCalls": bad_limit,  # type: ignore[typeddict-item]
                },
                pipeline,
            )
    with pytest.raises(ValueError, match="positive integer"):
        await execute_tool_batch(
            {"signal": AbortSignal(), "calls": []}, pipeline, max_parallel=0
        )

    # log without agent.session raises ValueError (wrapped in ToolBatchExecutionError)
    with pytest.raises(ToolBatchExecutionError):
        await execute_tool_batch(
            {
                "signal": AbortSignal(),
                "calls": [
                    {
                        "type": "tool-call",
                        "id": "log_fail",
                        "name": "proj_none",
                        "arguments": {},
                    }
                ],
                "log": {"turn": 1, "step": 1},
                "agent": {},
            },
            pipeline,
        )

    # Durable Session logging for completed + mid-batch skipped calls
    session = Session("s1")
    logged_agent = SimpleNamespace(session=session)
    mid_sig = AbortSignal()
    ctx.tools.register(
        define_tool(
            {
                "name": "abort_next",
                "description": "abort batch after first",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                    "presentationMeta": lambda _a, _v: {"k": "v"},
                },
                "execute": lambda _a, _c: (mid_sig.abort("stop rest"), "first")[1],
            }
        )
    )
    logged_batch = await execute_tool_batch(
        {
            "signal": mid_sig,
            "agent": logged_agent,
            "log": {"turn": 1, "step": 2},
            "maxParallelToolCalls": 2,
            "calls": [
                {
                    "type": "tool-call",
                    "id": "lg1",
                    "name": "abort_next",
                    "arguments": "{}",
                },
                {
                    "type": "tool-call",
                    "id": "lg2",
                    "name": "proj_none",
                    "arguments": "{}",
                },
            ],
        },
        pipeline,
    )
    assert len(logged_batch["results"]) == 2
    assert logged_batch["results"][0]["result"]["error"]["info"]["code"] == TOOL_ABORTED
    assert (
        logged_batch["results"][1]["result"]["error"]["info"]["code"]
        == TOOL_ABORTED_BEFORE_DISPATCH
    )
    logged_types = [ev.type for ev in session.events]
    assert logged_types == ["tool/call", "tool/result", "tool/call", "tool/result"]
    assert session.events[1].payload["error"]["code"] == TOOL_ABORTED
    assert session.events[3].payload["error"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH

    # Pre-aborted parallel group skips every call and logs cleanly
    session2 = Session("s2")
    pre_batch_sig = AbortSignal()
    pre_batch_sig.abort("already")
    ctx.tools.register(
        define_tool(
            {
                "name": "par_ok",
                "description": "par",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "isConcurrencySafe": lambda _a: True,
                "execute": lambda _a, _c: "par",
            }
        )
    )
    skipped_batch = await execute_tool_batch(
        {
            "signal": pre_batch_sig,
            "agent": SimpleNamespace(session=session2),
            "log": {"turn": 2, "step": 1},
            "calls": [
                {"type": "tool-call", "id": "sk1", "name": "par_ok", "arguments": "{}"},
                {"type": "tool-call", "id": "sk2", "name": "par_ok", "arguments": "{}"},
            ],
        },
        pipeline,
    )
    assert all(
        entry["result"]["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
        for entry in skipped_batch["results"]
    )
    assert [ev.type for ev in session2.events] == [
        "tool/call",
        "tool/result",
        "tool/call",
        "tool/result",
    ]

    # Mid-parallel-group abort skips remaining slots in the same group
    par_mid_sig = AbortSignal()
    ctx.tools.register(
        define_tool(
            {
                "name": "par_abort",
                "description": "par abort",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "isConcurrencySafe": lambda _a: True,
                "execute": lambda _a, _c: (par_mid_sig.abort("mid"), "ok")[1],
            }
        )
    )
    mid_par_batch = await execute_tool_batch(
        {
            "signal": par_mid_sig,
            "maxParallelToolCalls": 1,
            "calls": [
                {
                    "type": "tool-call",
                    "id": "pm1",
                    "name": "par_abort",
                    "arguments": "{}",
                },
                {"type": "tool-call", "id": "pm2", "name": "par_ok", "arguments": "{}"},
            ],
        },
        pipeline,
    )
    assert (
        mid_par_batch["results"][0]["result"]["error"]["info"]["code"] == TOOL_ABORTED
    )
    assert (
        mid_par_batch["results"][1]["result"]["error"]["info"]["code"]
        == TOOL_ABORTED_BEFORE_DISPATCH
    )

    # Scheduler failure inside dispatch drains remaining in-flight tasks before raising ToolBatchExecutionError
    slow_done = False

    async def _slow_par(_a: Any, _c: Any) -> str:
        nonlocal slow_done
        await asyncio.sleep(0.02)
        slow_done = True
        return "finished"

    ctx.tools.register(
        define_tool(
            {
                "name": "slow_par",
                "description": "slow",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "isConcurrencySafe": lambda _a: True,
                "execute": _slow_par,
            }
        )
    )
    call_count = 0
    orig_dispatch = pipeline.dispatch

    async def _boom_second_dispatch(exec_ctx: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("dispatch crashed")
        return await orig_dispatch(exec_ctx)

    with (
        patch.object(pipeline, "dispatch", side_effect=_boom_second_dispatch),
        pytest.raises(ToolBatchExecutionError),
    ):
        await execute_tool_batch(
            {
                "signal": AbortSignal(),
                "calls": [
                    {
                        "type": "tool-call",
                        "id": "dr1",
                        "name": "slow_par",
                        "arguments": "{}",
                    },
                    {
                        "type": "tool-call",
                        "id": "dr2",
                        "name": "par_ok",
                        "arguments": "{}",
                    },
                    {
                        "type": "tool-call",
                        "id": "dr3",
                        "name": "par_ok",
                        "arguments": "{}",
                    },
                ],
            },
            pipeline,
        )
    assert slow_done is True


@pytest.mark.asyncio
async def test_pipeline_remaining_edge_branches() -> None:
    """Covers remaining ToolExecutionPipeline and batch scheduler edge branches."""
    ctx = await _setup()
    pipeline = ToolExecutionPipeline(ctx)
    assert pipeline.ctx is ctx

    # _create_success_result: tuple render (732), failing render (727-728), failing presentationMeta (748-749)
    ctx.tools.register(
        define_tool(
            {
                "name": "tuple_render",
                "description": "tuple render",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: ({"type": "text", "text": v},),
                    "presentationMeta": lambda _a, v: {"val": v},
                },
                "execute": lambda _a, _c: "ok",
            }
        )
    )
    tr_res = await pipeline.run(
        {
            "callId": "tr1",
            "name": "tuple_render",
            "arguments": {},
            "signal": AbortSignal(),
        }
    )
    assert tr_res["isError"] is False
    assert tr_res["content"][0]["text"] == "ok"
    assert tr_res["meta"] == {"val": "ok"}

    ctx.tools.register(
        define_tool(
            {
                "name": "boom_render",
                "description": "boom render",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, _v: (_ for _ in ()).throw(
                        RuntimeError("render crash")
                    ),
                },
                "execute": lambda _a, _c: "ok",
            }
        )
    )
    assert (
        "output.render failed: render crash"
        in (
            await pipeline.run(
                {
                    "callId": "br1",
                    "name": "boom_render",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )

    ctx.tools.register(
        define_tool(
            {
                "name": "boom_meta",
                "description": "boom meta",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                    "presentationMeta": lambda _a, _v: (_ for _ in ()).throw(
                        RuntimeError("meta crash")
                    ),
                },
                "execute": lambda _a, _c: "ok",
            }
        )
    )
    assert (
        "output.presentationMeta failed: meta crash"
        in (
            await pipeline.run(
                {
                    "callId": "bm1",
                    "name": "boom_meta",
                    "arguments": {},
                    "signal": AbortSignal(),
                }
            )
        )["error"]["message"]
    )

    # _require_state with unregistered _ExecutionToken on another pipeline instance (336)
    prepared_tok = await pipeline.prepare(
        {
            "callId": "ut1",
            "name": "tuple_render",
            "arguments": {},
            "signal": AbortSignal(),
        }
    )
    assert prepared_tok["kind"] == "dispatch"
    foreign_pipeline = ToolExecutionPipeline(ctx)
    foreign_res = await foreign_pipeline.dispatch(prepared_tok["exec"])
    assert "missing execution state" in foreign_res["result"]["error"]["message"]

    # _service_ask where caller signal aborts AND approval returns "cancelled" (486)
    ask_abort_sig = AbortSignal()
    off_ask_cancel = ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "ask", "prompt": "confirm"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    ctx.provide(
        "approval",
        SimpleNamespace(
            request=lambda _r: (ask_abort_sig.abort("caller gone"), "cancelled")[1]
        ),
    )
    ask_cancel_res = await pipeline.run(
        {
            "callId": "ac1",
            "name": "tuple_render",
            "arguments": {},
            "agent": SimpleNamespace(),
            "signal": ask_abort_sig,
        }
    )
    assert ask_cancel_res["error"]["info"]["code"] == TOOL_ABORTED_BEFORE_DISPATCH
    off_ask_cancel()

    # _post_execute cleans up empty additionalContexts (877)
    off_empty_ctx = ctx.on(
        "tools/execute",
        lambda ex, next_fn: _wrap_empty_ctx(ex, next_fn),
        prepend=True,
        mode=EventMode.WATERFALL,
    )

    async def _wrap_empty_ctx(ex: Any, next_fn: Any) -> Any:
        res = await next_fn(ex)
        return {**res, "additionalContexts": []}

    empty_ctx_res = await pipeline.run(
        {
            "callId": "ec1",
            "name": "tuple_render",
            "arguments": {},
            "signal": AbortSignal(),
        }
    )
    assert "additionalContexts" not in empty_ctx_res
    off_empty_ctx()

    # complete() cancellation during tools/post-execute preserves prior additionalContexts (210, 908)
    post_abort_sig = AbortSignal()
    deferred_msg = create_user_message(content=[{"type": "text", "text": "keep ctx"}])
    ctx.tools.register(
        define_tool(
            {
                "name": "defer_then_post_abort",
                "description": "defer then post abort",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "execute": lambda _a, r: (r["deferContext"](deferred_msg), "done")[1],
            }
        )
    )

    async def _abort_in_post(_ex: Any, res: Any, _n: Any) -> Any:
        post_abort_sig.abort("post cancel")
        return {"kind": "accept"}

    off_post_abort = ctx.on(
        "tools/post-execute",
        _abort_in_post,
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    post_ab_res = await pipeline.run(
        {
            "callId": "pa1",
            "name": "defer_then_post_abort",
            "arguments": {},
            "signal": post_abort_sig,
        }
    )
    assert post_ab_res["error"]["info"]["code"] == TOOL_ABORTED
    assert post_ab_res["additionalContexts"] == [deferred_msg]
    off_post_abort()

    # BatchSettlement commit logs meta (1049) & error without info (1042->1048)
    session3 = Session("s3")
    ctx.tools.register(
        define_tool(
            {
                "name": "blocked_ok",
                "description": "ok then blocked",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [{"type": "text", "text": v}],
                },
                "execute": lambda _a, _c: "ok",
            }
        )
    )
    off_block_name = ctx.on(
        "tools/post-execute",
        lambda ex, _r, _n: (
            {"kind": "block", "feedback": [{"type": "text", "text": "policy blocked"}]}
            if ex["name"] == "blocked_ok"
            else {"kind": "accept"}
        ),
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    batch_s3 = await execute_tool_batch(
        {
            "signal": AbortSignal(),
            "agent": SimpleNamespace(session=session3),
            "log": {"turn": 3, "step": 1},
            "calls": [
                {
                    "type": "tool-call",
                    "id": "b1",
                    "name": "tuple_render",
                    "arguments": "{}",
                },
                {
                    "type": "tool-call",
                    "id": "b2",
                    "name": "blocked_ok",
                    "arguments": "{}",
                },
            ],
        },
        pipeline,
    )
    assert batch_s3["results"][0]["result"]["meta"] == {"val": "ok"}
    assert batch_s3["results"][1]["result"]["error"] == {"message": "policy blocked"}
    assert session3.events[1].payload["meta"] == {"val": "ok"}
    assert "error" not in session3.events[3].payload
    off_block_name()

    # _fill_pool detects abort inside pre-execute (1170) & assert_never on bogus prepare result (1148)
    fill_abort_sig = AbortSignal()

    async def _abort_on_pre(_ex: Any, _n: Any) -> Any:
        fill_abort_sig.abort("pre abort")
        return {"kind": "allow"}

    off_pre_abort = ctx.on(
        "tools/pre-execute",
        _abort_on_pre,
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    pre_ab_batch = await execute_tool_batch(
        {
            "signal": fill_abort_sig,
            "calls": [
                {
                    "type": "tool-call",
                    "id": "f1",
                    "name": "tuple_render",
                    "arguments": "{}",
                },
                {
                    "type": "tool-call",
                    "id": "f2",
                    "name": "tuple_render",
                    "arguments": "{}",
                },
            ],
        },
        pipeline,
    )
    assert len(pre_ab_batch["results"]) == 2
    assert (
        pre_ab_batch["results"][1]["result"]["error"]["info"]["code"]
        == TOOL_ABORTED_BEFORE_DISPATCH
    )
    off_pre_abort()

    async def _bogus_prepare(_ex: Any) -> dict[str, Any]:
        return {"kind": "bogus"}

    with (
        patch.object(pipeline, "prepare", side_effect=_bogus_prepare),
        pytest.raises(ToolBatchExecutionError) as exc_info,
    ):
        await execute_tool_batch(
            {
                "signal": AbortSignal(),
                "calls": [
                    {
                        "type": "tool-call",
                        "id": "bg1",
                        "name": "tuple_render",
                        "arguments": "{}",
                    }
                ],
            },
            pipeline,
        )
    assert "prepare result" in str(exc_info.value.__cause__)
