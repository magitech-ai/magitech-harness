"""Unit tests for ReactLoopAgent turn/step execution and projections."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import pytest
from anima import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.agent.model_selection import model_selection_projection_definition
from magitech_harness.agent.react_loop import ReactLoopAgent
from magitech_harness.agent.registry import AgentRegistry
from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.message import create_user_message
from magitech_harness.llm.retry import apply as apply_llm_retry
from magitech_harness.llm.service import LlmAdapter, LLMService
from magitech_harness.llm.types import GenerateOptions, LlmModelInfo, StreamChunk
from magitech_harness.llm.values import is_frozen
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.contracts import (
    ToolActivationError,
    ToolBatchExecutionError,
)
from magitech_harness.tools.native_executor import apply as apply_native_executor
from magitech_harness.tools.service import ToolService


class _StubAdapter(LlmAdapter):
    def __init__(self, handler: Any) -> None:
        self._handler = handler
        self.calls: list[GenerateOptions] = []

    def list_models(self, provider: str) -> list[LlmModelInfo]:
        return [
            {
                "id": "stub-model",
                "name": "Stub",
                "provider": provider,
                "context": {"contextWindow": 8192},
                "defaultMaxTokens": 1024,
                "reasoning": {
                    "defaultEffort": "medium",
                    "efforts": [
                        {"id": "low", "name": "Low"},
                        {"id": "medium", "name": "Medium"},
                    ],
                },
            }
        ]

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        self.calls.append(options)
        async for chunk in self._handler(options):
            yield chunk


async def _mount_runtime(
    handler: Any,
    *,
    register_adapter: bool = True,
) -> tuple[Context, _StubAdapter, Any, Any]:
    ctx = Context()
    ctx.use(SystemPromptService, {"include_identity": True, "persona_prefix": "PRE"})
    ctx.use(ToolService)
    ctx.use(apply_native_executor)
    ctx.use(LLMService)
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    ctx.use(AgentRegistry)
    ctx.use(apply_llm_retry)
    await ctx.flush()

    from magitech_harness.agent.inbox import inbox_projection_definition

    model_handle = ctx.sessionProjections.register(
        model_selection_projection_definition
    )
    inbox_handle = ctx.sessionProjections.register(inbox_projection_definition)

    adapter = _StubAdapter(handler)
    if register_adapter:
        ctx.llm.register_adapter(["stub"], adapter)
    return ctx, adapter, model_handle, inbox_handle


def _user(text: str) -> dict[str, Any]:
    return dict(
        create_user_message(
            content=[{"type": "text", "text": text}], source={"kind": "user"}
        )
    )


@pytest.mark.asyncio
async def test_react_loop_single_turn_stream_and_frozen_request() -> None:
    observed_requests: list[Mapping[str, Any]] = []

    async def handler(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        observed_requests.append(options)
        assert is_frozen(options["messages"])
        yield {"type": "text-delta", "index": 0, "text": "Hi there"}
        yield {"type": "usage", "usage": {"inputTokens": 3, "outputTokens": 2}}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx, adapter, model_h, inbox_h = await _mount_runtime(handler)
    session = ctx.sessions.create("rl-1")
    agent = ReactLoopAgent(
        ctx,
        "rl-1",
        {"provider": "stub", "model": "stub-model"},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )
    statuses: list[str] = []
    frames: list[dict[str, Any]] = []
    agent.ctx.on("agent/status", lambda payload: statuses.append(payload["status"]))
    agent.ctx.on(
        "agent/assistant-stream", lambda payload: frames.append(payload["frame"])
    )

    agent.followup(_user("Hello"))
    await agent.when_idle()

    assert statuses == ["running", "idle"]
    assert len(adapter.calls) == 1
    assert observed_requests[0]["sessionId"] == "rl-1"
    assert [f["type"] for f in frames] == ["start", "chunk", "chunk", "chunk", "end"]
    assert frames[-1]["outcome"]["kind"] == "committed"
    assert frames[-1]["outcome"]["eventType"] == "assistant/message"
    types = [ev.type for ev in session.events]
    assert "turn/start" in types
    assert "system/message" in types
    assert "user/message" in types
    assert "request/header" in types
    assert "request/context" in types
    assert "assistant/message" in types
    assert types[-1] == "turn/end"
    assert session.events[-1].payload["reason"] == {"kind": "completed"}


@pytest.mark.asyncio
async def test_react_loop_tool_calls_max_parallel_additional_context_and_registry_diff() -> (
    None
):
    call_count = 0

    async def handler(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {
                "type": "tool-call-delta",
                "index": 0,
                "id": "call-1",
                "name": "ping",
                "argumentsDelta": '{"value":"a"}',
            }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
        else:
            yield {"type": "text-delta", "index": 0, "text": "Done"}
            yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx, _adapter, model_h, inbox_h = await _mount_runtime(handler)
    active_phase = [True]

    ctx.tools.register(
        {
            "name": "ping",
            "description": "Ping",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            "output": {
                "schema": {"type": "string"},
                "render": lambda _args, val: [{"type": "text", "text": val}],
            },
            "execute": lambda args, _tc: f"pong:{args['value']}",
        }
    )
    ctx.tools.register(
        {
            "name": "extra",
            "description": "Extra",
            "parameters": {"type": "object", "properties": {}},
            "output": {
                "schema": {"type": "string"},
                "render": lambda _args, val: [{"type": "text", "text": val}],
            },
            "execute": lambda _args, _tc: "",
        },
        {"when": lambda _c: active_phase[0]},
    )

    session = ctx.sessions.create("rl-tools")
    agent = ReactLoopAgent(
        ctx,
        "rl-tools",
        {"provider": "stub", "model": "stub-model"},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )

    async def _attach_extra_context(
        _exec_view: Any, _result: Any, next_fn: Any
    ) -> dict[str, Any]:
        res = await next_fn()
        return {**res, "additionalContexts": [_user("injected context")]}

    agent.ctx.on("tools/post-execute", _attach_extra_context)

    # Toggle extra tool off after step 1 completes
    def _on_event(_s: Any, ev: Any) -> None:
        if ev.type == "tool/result":
            active_phase[0] = False

    agent.ctx.on("session/event", _on_event)

    agent.followup(_user("Run ping"))
    await agent.when_idle()

    assert call_count == 2
    dev_events = [ev for ev in session.events if ev.type == "developer/message"]
    assert len(dev_events) == 1
    assert dev_events[0].payload["message"]["content"] == [
        {"type": "tool-removal", "toolName": "extra"}
    ]
    assert any(
        ev.type == "user/message"
        and ev.payload.get("content") == [{"type": "text", "text": "injected context"}]
        for ev in session.events
    )


@pytest.mark.asyncio
async def test_react_loop_request_error_retry_and_sticky_max_tokens() -> None:
    attempt = 0

    async def handler(_options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            yield {
                "type": "finish",
                "reason": {
                    "kind": "error",
                    "failure": {"message": "transient", "code": "TRANSIENT"},
                },
            }
            return
        if attempt == 2:
            yield {"type": "text-delta", "index": 0, "text": "Part 1"}
            yield {"type": "finish", "reason": {"kind": "max-tokens"}}
            return
        yield {"type": "text-delta", "index": 0, "text": "Part 2"}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx, _adapter, model_h, inbox_h = await _mount_runtime(handler)
    session = ctx.sessions.create("rl-retry")
    agent = ReactLoopAgent(
        ctx,
        "rl-retry",
        {"provider": "stub", "model": "stub-model", "maxTokens": 128},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )

    async def on_req_err(payload: dict[str, Any], _next: Any) -> dict[str, Any]:
        return {"kind": "retry"}

    async def on_stopping(payload: dict[str, Any]) -> None:
        if payload["turn"] == 1 and attempt == 2:
            agent.inject(_user("Continue"))

    agent.ctx.on("agent/request-error", on_req_err)
    agent.ctx.on("agent/turn-stopping", on_stopping)

    agent.followup(_user("Go"))
    await agent.when_idle()

    assert attempt == 3
    attempts = [ev for ev in session.events if ev.type == "assistant/attempt"]
    assert len(attempts) == 1
    assert session.events[-1].payload["reason"] == {"kind": "max-tokens"}


@pytest.mark.asyncio
async def test_react_loop_abort_preserves_interrupted_blocks_and_waking_after_abort() -> (
    None
):
    gate = asyncio.Event()
    step_num = 0

    async def handler(_options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        nonlocal step_num
        step_num += 1
        if step_num == 1:
            yield {"type": "text-delta", "index": 0, "text": "Partial answer"}
            await gate.wait()
            yield {"type": "finish", "reason": {"kind": "stop"}}
        else:
            yield {"type": "text-delta", "index": 0, "text": "Fresh turn"}
            yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx, _adapter, model_h, inbox_h = await _mount_runtime(handler)
    session = ctx.sessions.create("rl-abort")
    agent = ReactLoopAgent(
        ctx,
        "rl-abort",
        {"provider": "stub", "model": "stub-model"},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )

    stream_started = asyncio.Event()
    agent.ctx.on(
        "agent/assistant-stream",
        lambda p: stream_started.set() if p["frame"]["type"] == "chunk" else None,
    )
    agent.followup(_user("First"))
    await stream_started.wait()

    agent.cancel({"kind": "user"}, {"keepInbox": True})
    # Steer after abort must classify as next-turn and latch a wake
    agent.steer(_user("Wake next turn"))
    gate.set()
    await agent.when_idle()

    interrupted = [
        ev
        for ev in session.events
        if ev.type == "assistant/message" and ev.payload.get("interrupted") is True
    ]
    assert len(interrupted) == 1
    turn_ends = [ev.payload for ev in session.events if ev.type == "turn/end"]
    assert turn_ends[0]["reason"] == {"kind": "aborted", "reason": {"kind": "user"}}
    assert turn_ends[1]["reason"] == {"kind": "completed"}


@pytest.mark.asyncio
async def test_react_loop_pre_step_reject_and_no_adapter_middleware() -> None:
    ctx, _adapter, model_h, inbox_h = await _mount_runtime(
        lambda _o: None, register_adapter=False
    )
    session = ctx.sessions.create("rl-reject")
    agent = ReactLoopAgent(
        ctx,
        "rl-reject",
        {"provider": "custom", "model": "custom-model"},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )

    async def reject_first(payload: dict[str, Any], next_fn: Any) -> dict[str, Any]:
        if payload["turn"] == 1:
            return {"kind": "reject"}
        return await next_fn()

    agent.ctx.on("agent/pre-step", reject_first)
    agent.followup(_user("Blocked"))
    await agent.when_idle()
    assert session.events[-1].payload["reason"] == {"kind": "blocked"}

    # Turn 2: middleware handles unregistered llm/stream directly
    async def custom_stream(_opts: Any, _next: Any) -> AsyncIterator[StreamChunk]:
        yield {"type": "text-delta", "index": 0, "text": "via middleware"}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx.on("llm/stream", lambda opts, _next: custom_stream(opts, _next))
    agent.followup(_user("Allowed"))
    await agent.when_idle()
    assert session.events[-1].payload["reason"] == {"kind": "completed"}


@pytest.mark.asyncio
async def test_react_loop_maintenance_and_agent_request_guard() -> None:
    async def handler(_options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        yield {"type": "text-delta", "index": 0, "text": "ok"}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx, _adapter, model_h, inbox_h = await _mount_runtime(handler)
    session = ctx.sessions.create("rl-maint")
    agent = ReactLoopAgent(
        ctx,
        "rl-maint",
        {"provider": "stub", "model": "stub-model"},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    async def maint_job(_signal: AbortSignal) -> str:
        entered.set()
        await release.wait()
        return "cleaned"

    maint_task = asyncio.create_task(agent.run_maintenance(maint_job))
    await entered.wait()
    with pytest.raises(RuntimeError, match="already has active work"):
        await agent.run_maintenance(maint_job)

    # Latched wakeup fires once maintenance exits
    agent.followup(_user("after maintenance"))
    release.set()
    assert await maint_task == "cleaned"
    await agent.when_idle()
    assert session.events[-1].payload["reason"] == {"kind": "completed"}

    # Attempting to switch provider in agent/request fails loudly
    errors: list[Any] = []
    agent.ctx.on("agent/error", lambda p: errors.append(p["error"]))

    async def bad_req(payload: dict[str, Any], next_fn: Any) -> dict[str, Any]:
        cfg = await next_fn()
        return {**cfg, "provider": "other"}

    dispose_req = agent.ctx.on("agent/request", bad_req)
    agent.followup(_user("mutate route"))
    await agent.when_idle()
    dispose_req()
    assert len(errors) == 1
    assert "agent/request proposed" in str(errors[0])


@pytest.mark.asyncio
async def test_react_loop_edge_helpers_and_pre_step_guards() -> None:
    async def handler(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        yield {"type": "text-delta", "index": 0, "text": "ok"}
        yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx, _adapter, model_h, inbox_h = await _mount_runtime(handler)

    # Seed edge system/message shapes: empty [], multi-block, plus direct non-list check
    s_seed = ctx.sessions.create("rl-seed")
    for idx, payload in enumerate(
        (
            {
                "turn": 1,
                "step": 1,
                "message": {"id": "s-1", "role": "system", "content": []},
            },
            {
                "turn": 1,
                "step": 2,
                "message": {
                    "id": "s-2",
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "text", "text": "b"},
                    ],
                },
            },
        )
    ):
        s_seed.append("system/message", payload, surface_op="append")

    from magitech_harness.agent.react_loop import _system_nodes_of

    assert [n["text"] for n in _system_nodes_of(s_seed)] == ["", None]
    fake_ev = type(
        "Ev", (), {"type": "system/message", "payload": {"message": {"content": "raw"}}}
    )()
    fake_sess = type(
        "FakeSess",
        (),
        {
            "surface_nodes": lambda self: [0, 1],
            "event_at": lambda self, s: None if s == 0 else fake_ev,
        },
    )()
    assert _system_nodes_of(fake_sess) == [{"seq": 1, "text": None}]  # type: ignore[arg-type]

    # Agent without model selection fails loudly on _pre_step
    no_model = ReactLoopAgent(
        ctx,
        "rl-nomodel",
        {},
        s_seed,
        {
            "modelSelection": model_h.reader_for(s_seed),
            "inbox": inbox_h.reader_for(s_seed),
        },
        0,
    )
    # when_idle while already idle is a no-op
    await no_model.when_idle()

    errs: list[Any] = []
    no_model.ctx.on("agent/error", lambda p: errs.append(p["error"]))
    with pytest.raises(RuntimeError, match="without driver reservation"):
        await no_model._turn()

    no_model.followup(_user("go"))
    await no_model.when_idle()
    assert any("has no provider/model" in str(e) for e in errs)

    # Normal agent testing pre-step empty accept on step 1 and pre-step error disposal
    session = ctx.sessions.create("rl-prestep")
    agent = ReactLoopAgent(
        ctx,
        "rl-prestep",
        {"provider": "stub", "model": "stub-model"},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )
    agent_errors: list[Any] = []
    agent.ctx.on("agent/error", lambda p: agent_errors.append(p["error"]))

    disposed_calls = [0]
    orig_resolve = ctx.tools.resolve

    async def _wrapped_resolve(*args: Any, **kwargs: Any) -> Any:
        res = await orig_resolve(*args, **kwargs)
        orig_dispose = res.dispose

        def _tracked_dispose() -> None:
            disposed_calls[0] += 1
            orig_dispose()

        res.dispose = _tracked_dispose
        return res

    ctx.tools.resolve = _wrapped_resolve  # type: ignore[method-assign]

    # Case 1: agent/pre-step throws ToolActivationError after tool_set resolved
    async def throw_activation(payload: dict[str, Any], next_fn: Any) -> dict[str, Any]:
        raise ToolActivationError("broken", "bad predicate")

    unsub = agent.ctx.on("agent/pre-step", throw_activation)
    agent.followup(_user("boom"))
    await agent.when_idle()
    unsub()
    assert disposed_calls[0] == 1
    assert any(isinstance(e, ToolActivationError) for e in agent_errors)
    assert (
        session.events[-1].payload["reason"]["error"]["code"]
        == "TOOL_ACTIVATION_FAILED"
    )

    # Case 2: step 1 pre-step returns empty messages -> completed immediately
    async def empty_accept(payload: dict[str, Any], next_fn: Any) -> dict[str, Any]:
        res = await next_fn()
        return {**res, "messages": []}

    unsub = agent.ctx.on("agent/pre-step", empty_accept)
    agent.followup(_user("ignored"))
    await agent.when_idle()
    unsub()
    assert session.events[-1].payload["reason"] == {"kind": "completed"}

    # Case 3: step 1 completes, next-step had a queued message that pre-step empties -> break
    step_seen = [0]

    async def drop_step2(payload: dict[str, Any], next_fn: Any) -> dict[str, Any]:
        res = await next_fn()
        step_seen[0] += 1
        if step_seen[0] == 2:
            return {**res, "messages": []}
        agent.inject(_user("queued step 2"))
        return res

    unsub = agent.ctx.on("agent/pre-step", drop_step2)
    agent.followup(_user("step 1"))
    await agent.when_idle()
    unsub()
    assert step_seen[0] == 2
    assert session.events[-1].payload["reason"] == {"kind": "completed"}


@pytest.mark.asyncio
async def test_react_loop_stream_settlement_replay_state_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mode = ["structured_error"]

    class _PrepAdapter(_StubAdapter):
        def list_models(self, provider: str) -> list[LlmModelInfo]:
            base = dict(super().list_models(provider)[0])
            base["context"] = {"contextWindow": 16384}
            if mode[0] in ("happy_series", "effort_mismatch"):
                base["systemPromptUpdate"] = "in-history"
                base["toolUpdate"] = "in-history"
            return [cast(LlmModelInfo, base)]

        async def prepare_call(
            self, provider: str, model: str, signal: AbortSignal | None = None
        ) -> dict[str, Any]:
            if mode[0] == "prepare_fail":
                raise LlmError("Bad config", "INVALID_ARGUMENT")

            async def _stream(opts: GenerateOptions) -> AsyncIterator[StreamChunk]:
                self.calls.append(opts)
                async for chunk in self._handler(opts):
                    yield chunk

            return {"model": self.list_models(provider)[0], "stream": _stream}

    async def handler(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        if mode[0] == "structured_error":
            yield {
                "type": "finish",
                "reason": {
                    "kind": "error",
                    "failure": {
                        "message": "Rate limit",
                        "code": "RATE_LIMIT",
                        "status": 429,
                        "providerRetryAfterMs": 2500,
                        "requestId": "req-99",
                        "offloadImages": 2,
                    },
                },
            }
        elif mode[0] == "pre_start_raise":
            raise RuntimeError("boom before stream starts")
            yield  # pragma: no cover
        elif mode[0] == "post_start_raise":
            yield {"type": "text-delta", "index": 0, "text": "half"}
            raise RuntimeError("boom mid-stream")
        elif mode[0] == "abort_empty":
            while True:
                yield {"type": "usage", "usage": {"inputTokens": 0, "outputTokens": 0}}
                await asyncio.sleep(0.01)
        elif mode[0] == "abort_with_replay":
            yield {"type": "text-delta", "index": 0, "text": "partial"}
            yield {"type": "usage", "usage": {"inputTokens": 3, "outputTokens": 2}}
            yield {
                "type": "finish",
                "reason": {"kind": "stop"},
                "replayState": {"response": {"opaque": "r1"}},
            }
            while True:
                yield {"type": "usage", "usage": {"inputTokens": 3, "outputTokens": 2}}
                await asyncio.sleep(0.01)
        elif mode[0] == "happy_series":
            yield {"type": "text-delta", "index": 0, "text": "done"}
            yield {
                "type": "finish",
                "reason": {"kind": "stop"},
                "replayState": {"response": {"opaque": "r2"}},
            }

    ctx = Context()
    ctx.use(SystemPromptService, {"include_identity": True, "persona_prefix": "PRE"})
    ctx.use(ToolService)
    ctx.use(apply_native_executor)
    ctx.use(LLMService)
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    ctx.use(AgentRegistry)
    await ctx.flush()

    from magitech_harness.agent.inbox import inbox_projection_definition

    model_h = ctx.sessionProjections.register(model_selection_projection_definition)
    inbox_h = ctx.sessionProjections.register(inbox_projection_definition)
    adapter = _PrepAdapter(handler)
    ctx.llm.register_adapter(["stub"], adapter)

    session = ctx.sessions.create("rl-stream-edge")
    agent = ReactLoopAgent(
        ctx,
        "rl-stream-edge",
        {"provider": "stub", "model": "stub-model"},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )
    errs: list[Any] = []
    agent.ctx.on("agent/error", lambda p: errs.append(p["error"]))

    # 1. Structured LLM error without retry handler
    agent.followup(_user("e1"))
    await agent.when_idle()
    assert isinstance(errs[-1], LlmError)
    assert errs[-1].failure.get("requestId") == "req-99"

    # 2. Error raised before live.start() (started == False -> no assistant/attempt)
    from magitech_harness.llm.service import PreparedLlmCall

    async def _prep_sync_fail(cfg: Any, _sig: AbortSignal) -> PreparedLlmCall:
        def _sync_fail(_opts: GenerateOptions) -> AsyncIterator[StreamChunk]:
            raise RuntimeError("boom before stream starts")

        return PreparedLlmCall(
            config=dict(cfg),
            context={"contextWindow": 16384},
            _dispatch_stream=_sync_fail,
            adapter_defaults=None,
            retry_policy=None,
        )

    monkeypatch.setattr(ctx.llm, "prepare_call", _prep_sync_fail)
    attempts_before = sum(1 for e in session.events if e.type == "assistant/attempt")
    agent.followup(_user("e2"))
    await agent.when_idle()
    monkeypatch.undo()
    assert (
        sum(1 for e in session.events if e.type == "assistant/attempt")
        == attempts_before
    )

    # 3. Exception raised by stream after live.start() -> settles assistant/attempt + settlement failure ExceptionGroup
    async def _prep_mid_fail(cfg: Any, _sig: AbortSignal) -> PreparedLlmCall:
        async def _mid_fail(_opts: GenerateOptions) -> AsyncIterator[StreamChunk]:
            yield {"type": "text-delta", "index": 0, "text": "half"}
            raise RuntimeError("boom after live start")

        return PreparedLlmCall(
            config=dict(cfg),
            context={"contextWindow": 16384},
            _dispatch_stream=_mid_fail,
            adapter_defaults=None,
            retry_policy=None,
        )

    monkeypatch.setattr(ctx.llm, "prepare_call", _prep_mid_fail)
    agent.followup(_user("e3"))
    await agent.when_idle()
    assert (
        sum(1 for e in session.events if e.type == "assistant/attempt")
        == attempts_before + 1
    )

    orig_sess_append = session.append

    def _fail_attempt_append(ev_type: str, payload: Any, **kwargs: Any) -> Any:
        if ev_type == "assistant/attempt":
            raise RuntimeError("attempt settle rejected")
        return orig_sess_append(ev_type, payload, **kwargs)

    monkeypatch.setattr(session, "append", _fail_attempt_append)
    agent.followup(_user("e3-settle-fail"))
    await agent.when_idle()
    monkeypatch.undo()
    assert any(isinstance(e, ExceptionGroup) for e in errs)

    # 4. Abort after live.start() with empty blocks -> settles assistant/attempt
    mode[0] = "abort_empty"
    seen_open = asyncio.Event()
    unsub_s = agent.ctx.on(
        "agent/assistant-stream",
        lambda p: seen_open.set() if p["frame"]["type"] == "chunk" else None,
    )
    agent.followup(_user("e4"))
    await seen_open.wait()
    agent.cancel({"kind": "hook", "reason": "stop-empty"})
    await agent.when_idle()
    unsub_s()
    assert session.events[-1].payload["reason"] == {
        "kind": "aborted",
        "reason": {"kind": "hook", "reason": "stop-empty"},
    }

    # 5. Abort after blocks + usage + replayState -> settles interrupted assistant/message
    mode[0] = "abort_with_replay"
    seen_replay = asyncio.Event()
    unsub_s = agent.ctx.on(
        "agent/assistant-stream",
        lambda p: (
            seen_replay.set()
            if p["frame"]["type"] == "chunk"
            and p["frame"]["chunk"].get("type") == "finish"
            else None
        ),
    )
    agent.followup(_user("e5"))
    await seen_replay.wait()
    await asyncio.sleep(0.01)
    agent.cancel({"kind": "weird"})  # falls back to {"kind": "user"}
    await agent.when_idle()
    unsub_s()
    interrupted_msgs = [
        e
        for e in session.events
        if e.type == "assistant/message" and e.payload.get("interrupted")
    ]
    assert len(interrupted_msgs) == 1
    assert interrupted_msgs[0].payload["message"]["source"]["replayState"] == {
        "response": {"opaque": "r1"}
    }
    assert "usage" in interrupted_msgs[0].payload

    # 6. Normal stream with startsRequestSeries=True, explicit & implicit runtime-context, tool addition, contextWindow, systemPromptUpdate, replayState
    mode[0] = "happy_series"
    rt_val = ["/tmp/ws"]
    ctx.system_prompt.context({"name": "cwd", "order": 1, "text": lambda _c: rt_val[0]})

    # Step 1: initial startsSeries=True + explicit runtime-context placeholder in messages
    async def with_series_and_rt(
        payload: dict[str, Any], next_fn: Any
    ) -> dict[str, Any]:
        res = await next_fn()
        rt_marker = {
            "role": "user",
            "content": [{"type": "text", "text": "x"}],
            "source": {"kind": "runtime-context"},
        }
        return {
            **res,
            "startsRequestSeries": True,
            "messages": [rt_marker, *res["messages"]],
        }

    # Use an agent with fresh session so header_logged starts False for startsSeries on initial
    s_happy = ctx.sessions.create("rl-happy-series")
    a_happy = ReactLoopAgent(
        ctx,
        "rl-happy-series",
        {"provider": "stub", "model": "stub-model"},
        s_happy,
        {
            "modelSelection": model_h.reader_for(s_happy),
            "inbox": inbox_h.reader_for(s_happy),
        },
        0,
    )
    unsub_ps = a_happy.ctx.on("agent/pre-step", with_series_and_rt)
    a_happy.followup(_user("turn 1"))
    await a_happy.when_idle()

    # Step 2: identical header + startsRequestSeries=True -> logs reason="series"
    rt_val[0] = "/tmp/ws2"
    # Swap back to default messages (context_msg appended implicitly when absent) + series
    unsub_ps()

    async def only_series(payload: dict[str, Any], next_fn: Any) -> dict[str, Any]:
        return {
            "kind": "enter",
            "messages": list(payload["messages"]),
            "startsRequestSeries": True,
        }

    unsub_ps = a_happy.ctx.on("agent/pre-step", only_series)
    a_happy.followup(_user("turn 2"))
    await a_happy.when_idle()

    # Turn 2b: config-only header change (no tool addition/removal) -> hits non-tool diff branch
    unsub_max = a_happy.ctx.on(
        "agent/request",
        lambda p, nxt: _set_max_tokens(p, nxt),
    )

    async def _set_max_tokens(_p: Any, nxt: Any) -> dict[str, Any]:
        cfg = await nxt()
        return {**cfg, "maxTokens": 256}

    a_happy.followup(_user("turn 2b"))
    await a_happy.when_idle()
    unsub_max()

    # Turn 3: register a new tool + startsRequestSeries -> header reason="change", startsSeries=True, tool-addition with headerSeq!
    ctx.tools.register(
        {
            "name": "added_tool",
            "description": "Added",
            "parameters": {"type": "object", "properties": {}},
            "output": {
                "schema": {"type": "string"},
                "render": lambda _a, v: [{"type": "text", "text": v}],
            },
            "execute": lambda _a, _tc: "ok",
        }
    )
    a_happy.followup(_user("turn 3"))
    await a_happy.when_idle()
    unsub_ps()

    headers = [e.payload for e in s_happy.events if e.type == "request/header"]
    assert headers[0].get("startsSeries") is True
    assert any(h.get("reason") == "series" for h in headers)
    assert any(
        h.get("reason") == "change" and h.get("startsSeries") is True for h in headers
    )
    dev_adds = [e.payload for e in s_happy.events if e.type == "developer/message"]
    assert len(dev_adds) == 1 and "headerSeq" in dev_adds[0]

    # 7. reasoningEffort mutation rejected; prepare non-NO_ADAPTER error propagates
    happy_errs: list[Any] = []
    a_happy.ctx.on("agent/error", lambda p: happy_errs.append(p["error"]))

    async def bad_effort(payload: dict[str, Any], next_fn: Any) -> dict[str, Any]:
        cfg = await next_fn()
        return {**cfg, "reasoningEffort": "high"}

    unsub_req = a_happy.ctx.on("agent/request", bad_effort)
    a_happy.followup(_user("effort fail"))
    await a_happy.when_idle()
    unsub_req()
    assert any("reasoning effort" in str(e) for e in happy_errs)

    mode[0] = "prepare_fail"
    a_happy.followup(_user("prep fail"))
    await a_happy.when_idle()
    assert any(
        isinstance(e, LlmError) and e.code == "INVALID_ARGUMENT" for e in happy_errs
    )


@pytest.mark.asyncio
async def test_react_loop_recovery_batch_errors_and_multi_turn_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_n = [0]
    hold_maint = asyncio.Event()

    async def handler(options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        call_n[0] += 1
        if call_n[0] in (1, 2, 3):
            yield {
                "type": "tool-call-delta",
                "index": 0,
                "id": f"c-{call_n[0]}",
                "name": "ping",
                "argumentsDelta": "{}",
            }
            yield {"type": "finish", "reason": {"kind": "tool-calls"}}
        else:
            yield {"type": "text-delta", "index": 0, "text": f"ok-{call_n[0]}"}
            yield {"type": "finish", "reason": {"kind": "stop"}}

    ctx, _adapter, model_h, inbox_h = await _mount_runtime(handler)
    ctx.tools.register(
        {
            "name": "ping",
            "description": "Ping",
            "parameters": {"type": "object", "properties": {}},
            "output": {
                "schema": {"type": "string"},
                "render": lambda _a, v: [{"type": "text", "text": v}],
            },
            "execute": lambda _a, _tc: "pong",
        }
    )

    session = ctx.sessions.create("rl-rec")
    agent = ReactLoopAgent(
        ctx,
        "rl-rec",
        {"provider": "stub", "model": "stub-model"},
        session,
        {
            "modelSelection": model_h.reader_for(session),
            "inbox": inbox_h.reader_for(session),
        },
        0,
    )

    # max_parallel_tool_calls picks up callable or int from ctx.agentLoop
    ctx.provide(
        "agentLoop", type("StubLoop", (), {"max_parallel_tool_calls": lambda self: 4})()
    )
    assert agent._max_parallel_tool_calls() == 4

    errs: list[Any] = []
    agent.ctx.on("agent/error", lambda p: errs.append(p["error"]))

    # 1. ToolBatchExecutionError with partial additionalContexts + Exception cause -> recovery fills tool/result
    orig_exec = ctx.tools.execute
    exec_mode = ["err_exc"]
    other_session = ctx.sessions.create("rl-other-rec")

    async def _mock_exec(batch: Any, **kwargs: Any) -> Any:
        other_session.append("user/message", _user("unrelated"), surface_op="append")
        if exec_mode[0] == "err_exc":
            session.append(
                "tool/call",
                {
                    "turn": 1,
                    "step": 1,
                    "callId": "orphan-1",
                    "name": "ping",
                    "arguments": {},
                },
            )
            raise ToolBatchExecutionError(
                cause=RuntimeError("batch boom"),
                partial={"results": [], "additionalContexts": [_user("partial ctx 1")]},
            )
        if exec_mode[0] == "err_str":
            raise ToolBatchExecutionError(
                cause="string cause boom",
                partial={"results": [], "additionalContexts": [_user("partial ctx 2")]},
            )
        return await orig_exec(batch, **kwargs)

    monkeypatch.setattr(ctx.tools, "execute", _mock_exec)
    agent.followup(_user("t1"))
    await agent.when_idle()
    assert any(isinstance(e, RuntimeError) and "batch boom" in str(e) for e in errs)
    assert any(ev.type == "tool/result" for ev in session.events)

    exec_mode[0] = "err_str"
    agent.followup(_user("t2"))
    await agent.when_idle()
    assert any("string cause boom" in str(e) for e in errs)

    # 2. Recovery append failure raises ExceptionGroup, AND turn/end append failure propagates through _throw_error
    exec_mode[0] = "err_exc"
    orig_append = session.append

    def _fail_rec_and_end(ev_type: str, payload: Any, **kwargs: Any) -> Any:
        if ev_type == "tool/result":
            raise RuntimeError("rec append failed")
        if ev_type == "turn/end":
            raise RuntimeError("end append failed")
        return orig_append(ev_type, payload, **kwargs)

    monkeypatch.setattr(session, "append", _fail_rec_and_end)
    agent.followup(_user("t3"))
    await agent.when_idle()
    monkeypatch.undo()
    assert any(isinstance(e, ExceptionGroup) for e in errs)
    assert any("end append failed" in str(e) for e in errs)

    # 3. Multi-turn: queue followup while turn 1 runs, and also test cancel keepInbox=False clearing wake_requested
    exec_mode[0] = "normal"
    monkeypatch.setattr(ctx.tools, "execute", orig_exec)

    # Turn 1 completes, second followup already waiting -> _turn returns True and runs Turn 2
    sent_extra = [False]

    def _queue_on_stopping(_p: Any) -> None:
        if not sent_extra[0]:
            sent_extra[0] = True
            agent.followup(_user("second turn queued"))

    unsub_ev = agent.ctx.on("agent/turn-stopping", _queue_on_stopping)
    agent.followup(_user("first turn"))
    await agent.when_idle()
    unsub_ev()
    assert call_n[0] >= 5

    # 4. Abort maintenance with disposed suppresses pending wake
    in_maint = asyncio.Event()

    async def _maint(_sig: AbortSignal) -> None:
        in_maint.set()
        await hold_maint.wait()

    m_task = asyncio.create_task(agent.run_maintenance(_maint))
    await in_maint.wait()
    agent.followup(_user("wont wake after dispose"))
    agent.cancel({"kind": "disposed"}, {"keepInbox": True})
    hold_maint.set()
    await m_task
    assert agent.inbox.has_pending

    # 5. Live cancel with keepInbox=False resets wake_requested
    agent.cancel({"kind": "user"}, {"keepInbox": False})
    assert not agent.inbox.has_pending

    # 6. turn/start append failure hits _throw_error cleanly
    def _fail_turn_start(ev_type: str, payload: Any, **kwargs: Any) -> Any:
        if ev_type == "turn/start":
            raise RuntimeError("turn/start fail")
        return orig_append(ev_type, payload, **kwargs)

    monkeypatch.setattr(session, "append", _fail_turn_start)
    agent.followup(_user("fail start"))
    await agent.when_idle()
    monkeypatch.undo()
    assert any("turn/start fail" in str(e) for e in errs)

    # 7. Non-mapping abort reason + driver task cancellation BaseException path
    from magitech_harness.agent.cancellation import AbortController
    from magitech_harness.agent.react_loop import _aborted_cancel_cause

    raw_ac = AbortController()
    raw_ac.abort("string-abort")
    assert _aborted_cancel_cause(raw_ac.signal) == {"kind": "user"}

    from magitech_harness.agent.react_loop import _llm_error_from_failure

    minimal_err = _llm_error_from_failure({"message": "plain", "code": "PLAIN_ERR"})
    assert minimal_err.code == "PLAIN_ERR"
    assert "status" not in minimal_err.failure

    # Post-stream settlement failure before live.settle (create_assistant_message raises) abandons live stream
    import magitech_harness.agent.react_loop as _rl_mod

    abandoned_frames: list[Any] = []
    unsub_ab = agent.ctx.on(
        "agent/assistant-stream",
        lambda p: (
            abandoned_frames.append(p["frame"])
            if p["frame"].get("type") == "end"
            else None
        ),
    )
    with monkeypatch.context() as mp:
        mp.setattr(
            _rl_mod,
            "create_assistant_message",
            lambda **_kw: (_ for _ in ()).throw(RuntimeError("bad assistant msg")),
        )
        agent.followup(_user("bad-blocks"))
        await agent.when_idle()
    unsub_ab()
    assert any(f["outcome"]["kind"] == "abandoned" for f in abandoned_frames)

    entered_cancel = asyncio.Event()
    wait_forever = asyncio.Event()

    async def _hang_on_step(_p: Any, _nxt: Any) -> Any:
        entered_cancel.set()
        await wait_forever.wait()

    unsub_hang = agent.ctx.on("agent/pre-step", _hang_on_step)
    agent.followup(_user("cancel-task"))
    await entered_cancel.wait()
    next(iter(agent._driver_tasks)).cancel()
    with pytest.raises(asyncio.CancelledError):
        await agent.when_idle()
    unsub_hang()

    # Legacy function-envelope tool schema in baseline request/header diffs without KeyError
    agent.session.append(
        "request/header",
        {
            "header": {
                "config": {"provider": "stub", "model": "stub-model"},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "legacy_tool",
                            "description": "Legacy",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            "reason": "initial",
        },
    )
    agent.followup(_user("after-legacy-header"))
    await agent.when_idle()
    assert any(
        ev.type == "developer/message"
        and any(
            part.get("type") == "tool-removal" and part.get("toolName") == "legacy_tool"
            for part in ev.payload["message"]["content"]
        )
        for ev in agent.session.events
    )
