"""Unit tests for ToolsService (including legacy ToolService transition coverage)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from anima import Context, EventMode

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.scope import bind_scope_parent, create_scope
from magitech_harness.session.events import ToolCallData, ToolResultData
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.contracts import (
    RUN_CODE_NAME,
    TOOL_ABORTED_BEFORE_DISPATCH,
    ToolActivationError,
    ToolDefinition,
    ToolExecutorUnavailableError,
)
from magitech_harness.tools.definition import define_tool as legacy_define_tool
from magitech_harness.tools.native_executor import apply as apply_native_executor
from magitech_harness.tools.schema import define_tool
from magitech_harness.tools.service import ToolRuntime, ToolService, ToolsService


def _parity_tool(
    name: str = "echo",
    reply: str | None = None,
    **overrides: Any,
) -> ToolDefinition:
    spec: dict[str, Any] = {
        "name": name,
        "description": f"tool {name}",
        "parameters": {"text": {"type": "string"}},
        "output": {
            "schema": {"type": "string"},
            "render": lambda _a, val: [{"type": "text", "text": val}],
        },
        "execute": (
            (lambda a, _c: a.get("text", ""))
            if reply is None
            else (lambda _a, _c: reply)
        ),
    }
    spec.update(overrides)
    return define_tool(spec)


async def _mount_tools() -> Context:
    ctx = Context()
    ctx.use(SystemPromptService)
    ctx.use(ToolsService)
    ctx.use(apply_native_executor)
    await ctx.flush()
    return ctx


# ---------------------------------------------------------------------------
# Legacy ToolService compatibility tests (kept active until P3-6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_service_register_unregister_and_get() -> None:
    """Verifies legacy tool registration, retrieval, schema export, and disposal."""
    ctx = Context()
    service = ToolService(ctx)

    t1 = legacy_define_tool(name="calc", description="calculator", execute=lambda a: 1)
    disposer = service.register(t1)

    assert service.get("calc") is t1
    assert len(service.get_schemas()) == 1
    assert service.get_schemas()[0]["function"]["name"] == "calc"

    disposer()
    assert service.get("calc") is None
    assert service.get_schemas() == []
    disposer()


@pytest.mark.asyncio
async def test_tool_service_tools_property_returns_copy() -> None:
    """Verifies the legacy tools mapping is a defensive copy."""
    ctx = Context()
    service = ToolService(ctx)

    tool = legacy_define_tool(
        name="calc", description="calculator", execute=lambda a: 1
    )
    service.register(tool)

    mapping = service.tools
    assert mapping == {"calc": tool}
    mapping["spoof"] = tool
    mapping.clear()
    assert service.get("calc") is tool
    assert service.tools == {"calc": tool}


@pytest.mark.asyncio
async def test_tool_service_unknown_tool_legacy() -> None:
    """Legacy execution of nonexistent tool returns error result and emits event."""
    ctx = Context()
    service = ToolService(ctx)
    results_emitted: list[tuple[str, bool]] = []
    ctx.on(
        "tools/result",
        lambda call, res: results_emitted.append((call.name, res.is_error)),
    )

    call_data = ToolCallData(call_id="call_x", name="nonexistent", arguments={})
    res = await service.execute(call_data)

    assert res.is_error
    assert "not found" in res.content
    assert results_emitted == [("nonexistent", True)]


@pytest.mark.asyncio
async def test_tool_service_execute_exception_handling_legacy() -> None:
    """Legacy exceptions during tool execution trap into ToolResultData error."""
    ctx = Context()
    service = ToolService(ctx)

    def crashing_tool(_args: Any) -> Any:
        raise ValueError("Division by zero in tool")

    service.register(
        legacy_define_tool(name="crash", description="Crashes", execute=crashing_tool)
    )

    res = await service.execute(
        ToolCallData(call_id="c_err", name="crash", arguments={})
    )
    assert res.is_error
    assert "Division by zero" in res.content


@pytest.mark.asyncio
async def test_tool_service_waterfall_pre_and_post_execute_legacy() -> None:
    """Legacy pre-execute and post-execute waterfall hooks."""
    ctx = Context()
    service = ToolService(ctx)
    service.register(
        legacy_define_tool(
            name="echo",
            description="echoes input",
            execute=lambda a: a.get("msg", ""),
        )
    )

    async def pre_hook(call: ToolCallData, next_fn: Any) -> Any:
        rewritten = ToolCallData(
            call_id=call.call_id,
            name=call.name,
            arguments={"msg": f"pre_{call.arguments.get('msg')}"},
        )
        return await next_fn(rewritten)

    async def post_hook(raw_result: str, _call: ToolCallData, next_fn: Any) -> Any:
        return await next_fn(f"post_{raw_result}")

    ctx.on("tools/pre-execute", pre_hook, mode=EventMode.WATERFALL)
    ctx.on("tools/post-execute", post_hook, mode=EventMode.WATERFALL)

    res = await service.execute(
        ToolCallData(call_id="c1", name="echo", arguments={"msg": "hello"})
    )
    assert not res.is_error
    assert res.content == "post_pre_hello"


# ---------------------------------------------------------------------------
# 0.2.2-rc.1 ToolsService parity tests
# ---------------------------------------------------------------------------


def test_tool_runtime_alias_matches() -> None:
    """ToolRuntime and ToolService alias ToolsService."""
    assert ToolService is ToolsService
    assert ToolRuntime is ToolsService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_cfg",
    [{"mode": "ptc"}, {"maxParallelSubCalls": 4}],
)
async def test_rejects_removed_config_fields(bad_cfg: dict[str, Any]) -> None:
    """Rejects removed config fields during construction."""
    ctx = Context()
    ctx.use(SystemPromptService)
    await ctx.flush()
    with pytest.raises(ValueError, match="invalid config"):
        ToolsService(ctx, bad_cfg)


@pytest.mark.asyncio
async def test_register_validation_rollback_and_system_prompt_bridge() -> None:
    """Validates output requirement, run_code ban, change rollback, and prompt bridge."""
    ctx = await _mount_tools()

    with pytest.raises(ValueError, match="run_code"):
        ctx.tools.register(_parity_tool(RUN_CODE_NAME))

    with pytest.raises(ValueError, match="output"):
        ctx.tools.register(
            {
                "name": "no-output",
                "description": "missing output",
                "parameters": {"type": "object", "properties": {}},
                "execute": lambda _a, _c: None,
            }
        )

    threw = False

    def _boom_change() -> None:
        nonlocal threw
        if not threw:
            threw = True
            raise RuntimeError("boom change listener")

    with (
        patch.object(ctx, "emit", side_effect=lambda *a, **kw: _boom_change()),
        pytest.raises(RuntimeError, match="boom change listener"),
    ):
        ctx.tools.register(_parity_tool("echo"))
    assert ctx.tools.get("echo") is None
    assert ctx.tools.schemas() == []

    dispose = ctx.tools.register(_parity_tool("echo", deferLoading=True))
    assert [s["name"] for s in ctx.tools.schemas()] == ["echo"]
    assert ctx.tools.schemas()[0]["deferLoading"] is True

    # Feeds systemPrompt.assemble() stripping host callbacks while keeping deferLoading
    assembly = await ctx.systemPrompt.assemble()
    assert [s["name"] for s in assembly["tools"]] == ["echo"]
    assert assembly["tools"][0]["deferLoading"] is True
    dispose()
    assert ctx.tools.get("echo") is None

    # Registry allows arbitrary JSON Schema in parameters (e.g. anyOf/pattern/$ref)
    arb_dispose = ctx.tools.register(
        {
            "name": "arb",
            "description": "arbitrary schema",
            "parameters": {
                "type": "object",
                "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            },
            "output": {"schema": {"type": "string"}, "render": lambda _a, v: []},
            "execute": lambda _a, _c: "ok",
        }
    )
    assert ctx.tools.get("arb") is not None
    arb_dispose()

    with pytest.raises(ValueError, match="presentationMeta"):
        ctx.tools.register(
            {
                "name": "bad_meta",
                "description": "d",
                "parameters": {},
                "output": {
                    "schema": {"type": "string"},
                    "render": lambda _a, v: [],
                    "presentationMeta": "not-callable",  # type: ignore[typeddict-item]
                },
                "execute": lambda _a, _c: "ok",
            }
        )


@pytest.mark.asyncio
async def test_scoped_registration_shadowing_and_restrictions() -> None:
    """Scope-local tools shadow global and survive own restrict({allow})."""
    ctx = await _mount_tools()
    key_a = {"id": "a"}
    key_b = {"id": "b"}
    scope_a = create_scope(ctx, key_a)
    scope_b = create_scope(ctx, key_b)

    ctx.tools.register(_parity_tool("bash", "global-bash"))
    ctx.tools.register(_parity_tool("read", "global-read"))
    scope_a.ctx.tools.register(_parity_tool("bash", "local-bash"))
    scope_a.ctx.tools.register(_parity_tool("capture", "local-capture"))

    assert sorted(t["name"] for t in ctx.tools.schemas()) == ["bash", "read"]
    assert sorted(t["name"] for t in ctx.tools.schemas(key_a)) == [
        "bash",
        "capture",
        "read",
    ]
    assert sorted(t["name"] for t in ctx.tools.schemas(key_b)) == ["bash", "read"]

    lift_allow = scope_a.ctx.tools.restrict({"allow": ["read"]})
    # Own registrations ('bash', 'capture') remain exempt from own filter
    assert sorted(t["name"] for t in ctx.tools.schemas(key_a)) == [
        "bash",
        "capture",
        "read",
    ]
    assert [t["name"] for t in ctx.tools.schemas(key_b)] == ["bash", "read"]

    with pytest.raises(ValueError, match="requires a scoped context"):
        ctx.tools.restrict({"deny": ["read"]})
    with pytest.raises(ValueError, match="no-op"):
        scope_b.ctx.tools.restrict({})
    with pytest.raises(ValueError, match="unknown global tool"):
        scope_a.ctx.tools.restrict({"allow": ["ghost"]})

    # Inherited parent scope restriction; child can further restrict names ancestor denied
    lift_allow()
    scope_a_deny = scope_a.ctx.tools.restrict({"deny": ["read"]})
    child_key = {"id": "child"}
    bind_scope_parent(child_key, key_a)
    child_scope = create_scope(ctx, child_key)
    child_scope.ctx.tools.restrict({"deny": ["read"]})
    child_scope.ctx.tools.restrict({"allow": ["bash", "capture"]})
    assert sorted(t["name"] for t in ctx.tools.schemas(child_key)) == [
        "bash",
        "capture",
    ]
    scope_a_deny()

    assert sorted(t["name"] for t in ctx.tools.schemas(key_a)) == [
        "bash",
        "capture",
        "read",
    ]


@pytest.mark.asyncio
async def test_activation_resolve_and_toolset_isolation() -> None:
    """Conditional registrations admit at resolve(), honor dispose(), and surface failures."""
    ctx = await _mount_tools()
    agent = {"id": "act"}
    enabled = False
    ctx.tools.register(_parity_tool("cond", "done"), {"when": lambda _c: enabled})

    unavail = await ctx.tools.resolve(
        {
            "agent": agent,
            "modelSelection": {"provider": "t", "model": "m"},
            "signal": AbortSignal(),
        }
    )
    assert ctx.tools.schemas(agent, unavail) == []

    enabled = True
    avail = await ctx.tools.resolve(
        {
            "agent": agent,
            "modelSelection": {"provider": "t", "model": "m"},
            "signal": AbortSignal(),
        }
    )
    assert [s["name"] for s in ctx.tools.schemas(agent, avail)] == ["cond"]
    res_ok = await ctx.tools.execute(
        {
            "calls": [
                {"type": "tool-call", "id": "c1", "name": "cond", "arguments": "{}"}
            ],
            "signal": AbortSignal(),
            "agent": agent,
            "toolSet": avail,
        }
    )
    assert res_ok["results"][0]["result"]["isError"] is False

    avail.dispose()
    assert ctx.tools.schemas(agent, avail) == []

    # Throwing condition -> ToolActivationError
    cause = RuntimeError("vault error")
    ctx.tools.register(
        _parity_tool("bad-cond"),
        {"when": lambda _c: (_ for _ in ()).throw(cause)},
    )
    with pytest.raises(ToolActivationError) as exc_info:
        await ctx.tools.resolve(
            {
                "agent": agent,
                "modelSelection": {"provider": "t", "model": "m"},
                "signal": AbortSignal(),
            }
        )
    assert exc_info.value.code == "TOOL_ACTIVATION_FAILED"
    assert exc_info.value.cause is cause

    # Aborted condition -> AbortError
    cond_ctx = await _mount_tools()
    sig = AbortSignal()
    gate = asyncio.Event()
    fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

    async def _wait_cond(_c: Any) -> bool:
        gate.set()
        return await fut

    cond_ctx.tools.register(_parity_tool("async-cond"), {"when": _wait_cond})
    pending = asyncio.create_task(
        cond_ctx.tools.resolve(
            {
                "agent": agent,
                "modelSelection": {"provider": "t", "model": "m"},
                "signal": sig,
            }
        )
    )
    await gate.wait()
    sig.abort("stop")
    fut.set_result(True)
    with pytest.raises(AbortError):
        await pending


@pytest.mark.asyncio
async def test_executor_unload_unavailability_and_guards() -> None:
    """Executor unload makes new calls fail without fallthrough; guards stay monotonic."""
    ctx = await _mount_tools()
    ctx.tools.register(_parity_tool("echo"))
    agent_a = {"id": "a"}
    scope_a = create_scope(ctx, agent_a)

    calls: list[str] = []
    added = False

    def _g1(exec_info: Any) -> str | None:
        nonlocal added
        calls.append("first")
        if not added:
            added = True
            scope_a.ctx.tools.guard(lambda _e: "late denial")
        return None

    scope_a.ctx.tools.guard(_g1)
    scope_a.ctx.on(
        "tools/pre-execute",
        lambda _e, _n: {"kind": "allow"},
        prepend=True,
        mode=EventMode.WATERFALL,
    )
    res = await ctx.tools.execute(
        {
            "calls": [
                {"type": "tool-call", "id": "g1", "name": "echo", "arguments": "{}"}
            ],
            "signal": AbortSignal(),
            "agent": agent_a,
        }
    )
    assert res["results"][0]["result"]["content"][0]["text"] == "Error: late denial"
    assert calls == ["first"]

    # Unloading custom scoped executor rejects subsequent execute() even if custom sections existed
    parent_exec = type(
        "_ParentExec",
        (),
        {
            "sections": [
                {
                    "name": "exec-static",
                    "order": 1,
                    "text": "static sec",
                    "interpolate": False,
                },
                {
                    "name": "exec-dyn",
                    "order": 2,
                    "text": lambda defs: f"dyn:{','.join(d['name'] for d in defs)}",
                },
            ],
            "describe": lambda _self, defs: {
                "schemas": [],
                "knownNames": [d["name"] for d in defs],
            },
            "execute": lambda _self, batch: asyncio.sleep(
                0, result={"results": [], "additionalContexts": [], "concluded": False}
            ),
        },
    )()
    child_exec = type(
        "_ChildExec",
        (),
        {
            "sections": (),
            "describe": lambda _self, defs: {"schemas": []},
            "execute": lambda _self, batch: asyncio.sleep(
                0, result={"results": [], "additionalContexts": [], "concluded": False}
            ),
        },
    )()
    dispose_parent_exec = scope_a.ctx.tools.set_executor(parent_exec)
    with pytest.raises(ValueError, match="Only one tool executor"):
        scope_a.ctx.tools.set_executor(parent_exec)
    child_agent = {"id": "child_exec"}
    bind_scope_parent(child_agent, agent_a)
    child_scope = create_scope(ctx, child_agent)
    dispose_child_exec = child_scope.ctx.tools.set_executor(child_exec)

    # Parent sees its executor section; child overriding executor causes parent section text -> ""
    asm_a = await ctx.systemPrompt.assemble({"scope": agent_a})
    asm_child = await ctx.systemPrompt.assemble({"scope": child_agent})
    assert any(s["text"] == "dyn:echo" for s in asm_a["sections"])
    assert any(
        s["name"] == "exec-dyn" and s["text"] == "" for s in asm_child["sections"]
    )

    dispose_child_exec()
    dispose_parent_exec()

    # Partial section registration failure inside set_executor cleans up earlier sections
    fail_key = {"id": "fail_sec"}
    fail_scope = create_scope(ctx, fail_key)
    failing_exec = type(
        "_FailSecExec",
        (),
        {
            "sections": [
                {"name": "ok-sec", "order": 1, "text": "ok"},
                {"name": "ok-sec", "order": 2, "text": "duplicate-name"},
            ],
            "describe": lambda _self, _defs: {"schemas": []},
            "execute": lambda _self, _b: asyncio.sleep(
                0, result={"results": [], "additionalContexts": [], "concluded": False}
            ),
        },
    )()
    with pytest.raises(ValueError, match="ok-sec"):
        fail_scope.ctx.tools.set_executor(failing_exec)
    asm_after_fail = await ctx.systemPrompt.assemble({"scope": fail_key})
    assert all(s["name"] != "ok-sec" for s in asm_after_fail["sections"])
    with pytest.raises(ToolExecutorUnavailableError):
        scope_a.ctx.tools.select_executor()
    with pytest.raises(ToolExecutorUnavailableError):
        await ctx.tools.execute(
            {
                "calls": [
                    {"type": "tool-call", "id": "u1", "name": "echo", "arguments": "{}"}
                ],
                "signal": AbortSignal(),
                "agent": agent_a,
            }
        )


@pytest.mark.asyncio
async def test_pre_aborted_batch_returns_aborted_before_dispatch() -> None:
    """Pre-aborted unowned batch settles every call with ABORTED_BEFORE_DISPATCH."""
    ctx = await _mount_tools()
    invoked = False

    def _body(_a: Any, _c: Any) -> str:
        nonlocal invoked
        invoked = True
        return "nope"

    ctx.tools.register(_parity_tool("echo", execute=_body))
    sig = AbortSignal()
    sig.abort()
    batch = await ctx.tools.execute(
        {
            "signal": sig,
            "calls": [
                {"type": "tool-call", "id": "1", "name": "echo", "arguments": "{}"},
                {"type": "tool-call", "id": "2", "name": "echo", "arguments": "{}"},
            ],
        }
    )
    assert invoked is False
    assert [r["result"]["error"]["info"]["code"] for r in batch["results"]] == [
        TOOL_ABORTED_BEFORE_DISPATCH,
        TOOL_ABORTED_BEFORE_DISPATCH,
    ]


@pytest.mark.asyncio
async def test_service_additional_validation_and_edge_branches() -> None:
    """Covers remaining register/restrict/guard/executable/execute/legacy edge branches."""
    # Unresolvable systemPrompt slot raises at init
    bad_ctx = Context()
    bad_ctx.reflect.provide("systemPrompt", None, lambda: False)
    with (
        patch.object(bad_ctx, "has", return_value=True),
        pytest.raises(RuntimeError, match="unresolvable"),
    ):
        ToolsService(bad_ctx)

    ctx = await _mount_tools()
    with pytest.raises(ValueError, match="non-empty string name"):
        ctx.tools.register(
            {
                "name": "",
                "description": "d",
                "parameters": {},
                "output": {"schema": {}, "render": lambda _a, _v: []},
                "execute": lambda _a, _c: None,
            }
        )
    with pytest.raises(ValueError, match="description"):
        ctx.tools.register(
            {
                "name": "x",
                "description": 123,
                "parameters": {},
                "output": {"schema": {}, "render": lambda _a, _v: []},
                "execute": lambda _a, _c: None,
            }
        )
    with pytest.raises(ValueError, match="callable execute"):
        ctx.tools.register(
            {
                "name": "x",
                "description": "d",
                "parameters": {},
                "output": {"schema": {}, "render": lambda _a, _v: []},
                "execute": "no",
            }
        )
    with pytest.raises(ValueError, match="when"):
        ctx.tools.register(_parity_tool("x"), {"when": "not-callable"})  # type: ignore[arg-type]

    # Global restrict & run_code restriction error
    agent_r = {"id": "r"}
    scope_r = create_scope(ctx, agent_r)
    with pytest.raises(ValueError, match="run_code"):
        scope_r.ctx.tools.restrict({"allow": [RUN_CODE_NAME]})

    # Global guard denial (including after a passing global guard) & passing scoped guard
    off_g1 = ctx.tools.guard(lambda _e: None)
    off_g2 = ctx.tools.guard(lambda _e: "global denied")
    assert ctx.tools.check_guards({"callId": "1"}, scope=agent_r) == "global denied"
    off_g1()
    off_g2()
    off_sg = scope_r.ctx.tools.guard(lambda _e: None)
    assert ctx.tools.check_guards({"callId": "1"}, scope=agent_r) is None
    off_sg()

    # Unknown name get / executable / unowned conditional / disposed toolSet
    off_timeout = ctx.tools.register(_parity_tool("with_timeout", timeoutMs=250))
    assert ctx.tools.get("with_timeout")["timeoutMs"] == 250
    off_timeout()
    assert ctx.tools.get("missing", scope=agent_r) is None
    assert ctx.tools.executable("missing", agent_r) is None
    dispose_cond = ctx.tools.register(
        _parity_tool("cond_only"), {"when": lambda _c: True}
    )
    assert ctx.tools.executable("cond_only", agent_r) is None
    ts = await ctx.tools.resolve(
        {"agent": agent_r, "modelSelection": {}, "signal": AbortSignal()}
    )
    assert ctx.tools.executable("cond_only", {"id": "other"}, ts) is None
    dispose_cond()

    # Tool unregistered mid-resolve does not re-appear in snapshot view
    dispose_dynamic = ctx.tools.register(_parity_tool("dynamic"))
    ts_dyn = await ctx.tools.resolve(
        {"agent": agent_r, "modelSelection": {}, "signal": AbortSignal()}
    )
    dispose_late = ctx.tools.register(_parity_tool("late_unconditional"))
    assert ctx.tools.get("dynamic", agent_r, ts_dyn) is not None
    assert ctx.tools.get("late_unconditional", agent_r, ts_dyn) is None
    assert ctx.tools.executable("late_unconditional", agent_r, ts_dyn) is None
    assert [d["name"] for d in ctx.tools.definitions(agent_r, ts_dyn)] == ["dynamic"]
    assert ctx.tools.get("dynamic", {"id": "other"}, ts_dyn) is None
    assert ctx.tools.get("dynamic", agent_r, "not-a-toolset") is None
    assert ctx.tools.executable("dynamic", agent_r, "not-a-toolset") is None
    assert ctx.tools.definitions(agent_r, "not-a-toolset") == []
    dispose_late()
    dispose_dynamic()
    assert ctx.tools.executable("dynamic", agent_r, ts_dyn) is None
    ts_dyn.dispose()
    assert ctx.tools.get("dynamic", agent_r, ts_dyn) is None

    # No executor anywhere in fresh context raises ToolExecutorUnavailableError; set_executor works without sp
    empty_ctx = Context()
    empty_tools = ToolsService(empty_ctx)
    with pytest.raises(ToolExecutorUnavailableError):
        empty_tools.select_executor()
    empty_tools.set_executor(
        type(
            "_MinExec",
            (),
            {
                "describe": lambda _s, _d: {"schemas": []},
                "execute": lambda _s, _b: asyncio.sleep(
                    0,
                    result={
                        "results": [],
                        "additionalContexts": [],
                        "concluded": False,
                    },
                ),
            },
        )()
    )

    # Scoped get() + legacy _unregister after overwrite + resolve mid-when replacement
    ctx.tools.register(_parity_tool("visible_p"))
    assert ctx.tools.get("visible_p") is not None

    leg = legacy_define_tool(name="leg", description="leg", execute=lambda _a: "raw")
    leg2 = legacy_define_tool(name="leg", description="leg2", execute=lambda _a: "raw2")
    off_leg1 = ctx.tools.register(leg)
    ctx.tools.register(leg2)
    off_leg1()
    assert ctx.tools.get("leg") is leg2

    # Replacing registration during async when condition excludes stale snapshot
    dispose_race: list[Any] = []

    async def _racing_when(_c: Any) -> bool:
        dispose_race[0]()
        return True

    dispose_race.append(
        ctx.tools.register(_parity_tool("raced"), {"when": _racing_when})
    )
    ts_raced = await ctx.tools.resolve(
        {"agent": agent_r, "modelSelection": {}, "signal": AbortSignal()}
    )
    assert [s["name"] for s in ctx.tools.schemas(agent_r, ts_raced)] != ["raced"]

    # Legacy ToolResultData passthrough from tools/execute waterfall
    ctx.on(
        "tools/execute",
        lambda _ex, _next: ToolResultData(
            call_id="l1", name="leg", content="direct", is_error=False
        ),
        mode=EventMode.WATERFALL,
    )
    res_leg = await ctx.tools.execute(
        ToolCallData(call_id="l1", name="leg", arguments={})
    )
    assert res_leg.content == "direct"
