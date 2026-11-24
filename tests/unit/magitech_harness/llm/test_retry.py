"""Tests for provider-routed LLM request recovery and retry projection."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from animas import Context, EventMode, FiberState
from animas.jsexpr import dump_yaml
from animas.loader import load_profile

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.retry import (
    RetryId,
    apply,
    llm_retry_projection,
    llmRetryProjection,
    local_delay,
    localDelay,
    retry_id,
    retry_policy_key,
    retry_state_key,
    retryId,
    retryPolicyKey,
    retryStateKey,
    validate_config,
)
from magitech_harness.llm.retry_policy import resolve_retry_policy
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    ProjectionHandle,
    SessionProjectionRegistry,
)
from magitech_harness.session.store import Session


def _make_failure(
    *,
    code: str = "SERVER",
    message: str = "temporary failure",
    provider_retry_after_ms: float | None = None,
) -> dict[str, Any]:
    failure: dict[str, Any] = {
        "code": code,
        "message": message,
        "retryable": code != "AUTH",
    }
    if provider_retry_after_ms is not None:
        failure["providerRetryAfterMs"] = provider_retry_after_ms
    return failure


class _FakeAgent:
    def __init__(self, session: Session) -> None:
        self.session = session


async def _mount_retry(ctx: Context, *, internals: dict[str, Any] | None = None) -> Any:
    if internals is None:
        fiber = ctx.use(apply)
        await fiber.load()
        return fiber

    async def _plugin(plugin_ctx: Context) -> None:
        await apply(plugin_ctx, {}, internals)

    _plugin.inject = apply.inject  # type: ignore[attr-defined]
    fiber = ctx.use(_plugin)
    await fiber.load()
    return fiber


def _start_retry(ctx: Context, *, internals: dict[str, Any] | None = None) -> Any:
    """Mount without awaiting: registration can be deliberately deferred."""
    if internals is None:
        return ctx.use(apply)

    async def _plugin(plugin_ctx: Context) -> None:
        await apply(plugin_ctx, {}, internals)

    _plugin.inject = apply.inject  # type: ignore[attr-defined]
    return ctx.use(_plugin)


def _setup_context() -> tuple[Context, SessionProjectionRegistry, Session]:
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    session = Session.create("retry-test-session")
    return ctx, registry, session


async def _dispatch_request_error(
    ctx: Context,
    *,
    session: Session,
    turn: int = 1,
    step: int = 1,
    provider: str = "mock",
    failure: dict[str, Any] | None = None,
    retry_policy: dict[str, Any] | None = None,
    signal: AbortSignal | None = None,
    use_agent_wrapper: bool = True,
) -> Any:
    payload: dict[str, Any] = {
        "turn": turn,
        "step": step,
        "provider": provider,
        "failure": failure if failure is not None else _make_failure(),
        "retryPolicy": retry_policy,
        "signal": signal or AbortSignal(),
    }
    if use_agent_wrapper:
        payload["agent"] = _FakeAgent(session)
    else:
        payload["session"] = session
    return await ctx.waterfall(
        "agent/request-error",
        payload,
        default_handler=lambda _payload: None,
    )


def test_brand_and_config_validation() -> None:
    """RetryId helpers and strict empty-config validator enforce boundaries."""
    rid = retry_id("retry-123")
    assert rid == "retry-123"
    assert RetryId("retry-123") == rid
    assert retryId("retry-123") == rid

    validate_config(None)
    validate_config({})

    with pytest.raises(TypeError, match="llm-retry: config must be a mapping"):
        validate_config("invalid")

    with pytest.raises(
        ValueError,
        match="llm-retry: retryPolicy belongs under each provider configuration",
    ):
        validate_config({"retryPolicy": {"mode": "normal"}})

    with pytest.raises(ValueError, match='llm-retry: unknown key "extra"'):
        validate_config({"extra": True})


def test_key_builders_and_local_delay() -> None:
    """Policy keys, state keys, and backoff delay calculations match contract."""
    assert localDelay is local_delay
    assert retryPolicyKey is retry_policy_key
    assert retryStateKey is retry_state_key

    p_normal_a = resolve_retry_policy(
        {
            "mode": "normal",
            "maxRetries": 3,
            "retryableCodes": ["TIMEOUT", "SERVER"],
            "backoff": {"initialDelayMs": 100, "maxDelayMs": 1000, "jitterRatio": 0},
        }
    )
    p_normal_b = resolve_retry_policy(
        {
            "mode": "normal",
            "maxRetries": 3,
            "retryableCodes": ["SERVER", "TIMEOUT"],
            "backoff": {"initialDelayMs": 100, "maxDelayMs": 1000, "jitterRatio": 0},
        }
    )
    assert retry_policy_key(p_normal_a) == retry_policy_key(p_normal_b)
    assert json.loads(retry_policy_key(p_normal_a)) == [
        "normal",
        3,
        ["SERVER", "TIMEOUT"],
        100,
        1000,
        0,
    ]

    p_always = resolve_retry_policy(
        {
            "mode": "always",
            "backoff": {"initialDelayMs": 50, "maxDelayMs": 500, "jitterRatio": 0.2},
        }
    )
    assert json.loads(retry_policy_key(p_always)) == ["always", 50, 500, 0.2]
    assert json.loads(retry_state_key("openai", "pk")) == ["openai", "pk"]

    # Deterministic exponential growth capped at maxDelayMs and overflow-safe
    assert local_delay(p_normal_a, 1, lambda: 0.5) == 100
    assert local_delay(p_normal_a, 2, lambda: 0.5) == 200
    assert local_delay(p_normal_a, 3, lambda: 0.5) == 400
    assert local_delay(p_normal_a, 4, lambda: 0.5) == 800
    assert local_delay(p_normal_a, 5, lambda: 0.5) == 1000
    assert local_delay(p_normal_a, 5000, lambda: 0.5) == 1000

    # Jitter bounds and clamping
    p_jitter = resolve_retry_policy(
        {
            "mode": "normal",
            "maxRetries": 2,
            "backoff": {"initialDelayMs": 100, "maxDelayMs": 110, "jitterRatio": 0.5},
        }
    )
    assert local_delay(p_jitter, 1, lambda: 0.0) == pytest.approx(50.0)
    assert local_delay(p_jitter, 1, lambda: 1.0) == pytest.approx(110.0)


def test_projection_definition_and_schema() -> None:
    """llmRetry projection validates schema, folds idempotently, and resets."""
    assert llmRetryProjection is llm_retry_projection
    assert llm_retry_projection.key == "llmRetry"
    assert llm_retry_projection.state_version == 1
    assert llm_retry_projection.wire is None

    state = llm_retry_projection.init({}, 0)
    assert state == {}

    validated = llm_retry_projection.state_schema(
        {"k1": {"retry": 2, "retryId": "r-1"}}
    )
    assert validated == {"k1": {"retry": 2, "retryId": "r-1"}}

    for bad_value in (
        "not-a-dict",
        {1: {"retry": 1, "retryId": "r-1"}},
        {"k1": "bad-entry"},
        {"k1": {"retry": 1}},
        {"k1": {"retry": True, "retryId": "r-1"}},
        {"k1": {"retry": -1, "retryId": "r-1"}},
        {"k1": {"retry": 1, "retryId": 123}},
    ):
        with pytest.raises(ValueError):
            llm_retry_projection.state_schema(bad_value)

    session = Session.create("proj-fold")
    user_ev = session.append(
        "user/message",
        {
            "role": "user",
            "id": "u1",
            "content": [{"type": "text", "text": "go"}],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )
    assert llm_retry_projection.apply(state, user_ev) is state

    retry_ev = session.append(
        "llm/retry",
        {
            "retryId": "rid-1",
            "turn": 1,
            "step": 1,
            "provider": "mock",
            "mode": "normal",
            "policyKey": "pk-1",
            "retry": 1,
            "maxRetries": 3,
            "delayMs": 10.5,
            "failure": _make_failure(),
        },
    )
    next_state = llm_retry_projection.apply(state, retry_ev)
    skey = retry_state_key("mock", "pk-1")
    assert next_state == {skey: {"retry": 1, "retryId": "rid-1"}}
    assert llm_retry_projection.apply(next_state, retry_ev) is next_state

    # step/start and turn/end both clear the active retry chain map
    start_ev = session.append("step/start", {"turn": 1, "step": 2})
    assert llm_retry_projection.apply(next_state, start_ev) == {}

    end_ev = session.append("turn/end", {"turn": 1})
    assert llm_retry_projection.apply(next_state, end_ev) == {}


async def test_apply_requires_session_projections() -> None:
    """Mounting without sessionProjections fails loud immediately."""
    ctx = Context()
    with pytest.raises(
        RuntimeError, match="llm-retry requires the sessionProjections service"
    ):
        await apply(ctx, {})


async def test_normal_policy_retry_chain_and_exhaustion() -> None:
    """Normal policy retries transient codes, reuses retryId, and delegates when exhausted."""
    ctx, registry, session = _setup_context()
    waits: list[float] = []

    async def fake_sleep(delay_ms: float, _signal: AbortSignal) -> None:
        waits.append(delay_ms)

    await _mount_retry(
        ctx,
        internals={"random": lambda: 0.5, "sleep": fake_sleep},
    )
    try:
        policy = resolve_retry_policy(
            {
                "mode": "normal",
                "maxRetries": 2,
                "retryableCodes": ["SERVER"],
                "backoff": {"initialDelayMs": 10, "maxDelayMs": 50, "jitterRatio": 0},
            }
        )

        # Missing retryPolicy delegates cleanly
        assert (
            await _dispatch_request_error(
                ctx, session=session, retry_policy=None, use_agent_wrapper=False
            )
            is None
        )

        # Missing session/agent target fails loud
        with pytest.raises(ValueError, match="session"):
            await ctx.waterfall(
                "agent/request-error",
                {
                    "turn": 1,
                    "step": 1,
                    "provider": "mock",
                    "failure": _make_failure(),
                    "retryPolicy": policy,
                    "signal": AbortSignal(),
                },
            )

        # Non-retryable code delegates directly
        downstream_calls = 0

        def downstream_marker(payload: dict[str, Any], next_fn: Any) -> Any:
            nonlocal downstream_calls
            downstream_calls += 1
            return {"kind": "abort", "downstream": payload["failure"]["code"]}

        dispose_downstream = ctx.on(
            "agent/request-error", downstream_marker, mode=EventMode.WATERFALL
        )
        res_auth = await _dispatch_request_error(
            ctx,
            session=session,
            provider="mock",
            failure=_make_failure(code="AUTH"),
            retry_policy=policy,
        )
        assert res_auth == {"kind": "abort", "downstream": "AUTH"}
        assert downstream_calls == 1
        assert [e.type for e in session.events] == []

        # Attempt 1 & Attempt 2 retry and reuse retryId
        res1 = await _dispatch_request_error(
            ctx, session=session, provider="mock", retry_policy=policy
        )
        res2 = await _dispatch_request_error(
            ctx, session=session, provider="mock", retry_policy=policy
        )
        assert res1 == {"kind": "retry"}
        assert res2 == {"kind": "retry"}
        assert waits == [10, 20]

        retry_events = [e for e in session.events if e.type == "llm/retry"]
        started_events = [e for e in session.events if e.type == "llm/retry-started"]
        assert len(retry_events) == 2
        assert len(started_events) == 2
        assert retry_events[0].payload["retryId"] == retry_events[1].payload["retryId"]
        assert retry_events[0].payload["retry"] == 1
        assert retry_events[1].payload["retry"] == 2
        assert retry_events[0].payload["maxRetries"] == 2
        assert set(started_events[0].payload.keys()) == {
            "retryId",
            "turn",
            "step",
            "retry",
        }

        # Exhausted at maxRetries delegates downstream
        res3 = await _dispatch_request_error(
            ctx, session=session, provider="mock", retry_policy=policy
        )
        assert res3 == {"kind": "abort", "downstream": "SERVER"}
        assert downstream_calls == 2

        # step/start clears chain so retries start over with fresh retryId
        session.append("step/start", {"turn": 1, "step": 2})
        res4 = await _dispatch_request_error(
            ctx, session=session, step=2, provider="mock", retry_policy=policy
        )
        assert res4 == {"kind": "retry"}
        retry_events_after = [e for e in session.events if e.type == "llm/retry"]
        assert len(retry_events_after) == 3
        assert retry_events_after[2].payload["retry"] == 1
        assert (
            retry_events_after[2].payload["retryId"]
            != retry_events_after[0].payload["retryId"]
        )
        dispose_downstream()
    finally:
        registry.dispose()
        await ctx.fiber.dispose()


async def test_provider_retry_after_and_always_mode() -> None:
    """Retry-After honors caps in normal vs always mode, and always ignores downstream errors."""
    ctx, registry, session = _setup_context()
    waits: list[float] = []

    async def fake_sleep(delay_ms: float, _signal: AbortSignal) -> None:
        waits.append(delay_ms)

    await _mount_retry(
        ctx,
        internals={"random": lambda: 0.5, "sleep": fake_sleep},
    )
    try:
        normal_policy = resolve_retry_policy(
            {
                "mode": "normal",
                "maxRetries": 5,
                "backoff": {"initialDelayMs": 15, "maxDelayMs": 100, "jitterRatio": 0},
            }
        )
        always_policy = resolve_retry_policy(
            {
                "mode": "always",
                "backoff": {"initialDelayMs": 25, "maxDelayMs": 100, "jitterRatio": 0},
            }
        )

        # In-cap providerRetryAfterMs is used directly
        assert await _dispatch_request_error(
            ctx,
            session=session,
            failure=_make_failure(provider_retry_after_ms=42),
            retry_policy=normal_policy,
        ) == {"kind": "retry"}
        # Zero providerRetryAfterMs falls back to local_delay
        assert await _dispatch_request_error(
            ctx,
            session=session,
            failure=_make_failure(provider_retry_after_ms=0),
            retry_policy=normal_policy,
        ) == {"kind": "retry"}
        # Over-cap providerRetryAfterMs in normal mode delegates (does not retry)
        assert (
            await _dispatch_request_error(
                ctx,
                session=session,
                failure=_make_failure(provider_retry_after_ms=500),
                retry_policy=normal_policy,
            )
            is None
        )

        # Over-cap providerRetryAfterMs in always mode falls back to local_delay
        # and does not include maxRetries on logged llm/retry event.
        assert await _dispatch_request_error(
            ctx,
            session=session,
            failure=_make_failure(code="AUTH", provider_retry_after_ms=500),
            retry_policy=always_policy,
        ) == {"kind": "retry"}
        always_retry_ev = [
            e
            for e in session.events
            if e.type == "llm/retry" and e.payload["mode"] == "always"
        ][-1]
        assert "maxRetries" not in always_retry_ev.payload
        assert waits == [42, 30, 25]

        # Downstream returning {kind: 'retry'} in always mode short-circuits without logging
        downstream_mode = "retry"

        async def downstream_handler(_payload: dict[str, Any], _next: Any) -> Any:
            if downstream_mode == "retry":
                return {"kind": "retry"}
            if downstream_mode == "boom":
                raise RuntimeError("downstream exploded")
            return None

        dispose_down = ctx.on(
            "agent/request-error", downstream_handler, mode=EventMode.WATERFALL
        )
        before_count = len(session.events)
        assert await _dispatch_request_error(
            ctx, session=session, failure=_make_failure(), retry_policy=always_policy
        ) == {"kind": "retry"}
        assert len(session.events) == before_count

        # Downstream throwing in always mode warns and falls back to always backoff
        downstream_mode = "boom"
        assert await _dispatch_request_error(
            ctx, session=session, failure=_make_failure(), retry_policy=always_policy
        ) == {"kind": "retry"}
        assert waits[-1] == 50
        dispose_down()
    finally:
        registry.dispose()
        await ctx.fiber.dispose()


async def test_cancellation_real_sleep_and_disposal() -> None:
    """Abort during wait, downstream settle, and plugin disposal behave cleanly."""
    ctx, registry, session = _setup_context()
    await _mount_retry(ctx)
    try:
        policy = resolve_retry_policy(
            {
                "mode": "normal",
                "maxRetries": 3,
                "backoff": {"initialDelayMs": 5, "maxDelayMs": 50, "jitterRatio": 0},
            }
        )
        # Real short sleep completes and emits retry-started
        assert await _dispatch_request_error(
            ctx, session=session, retry_policy=policy
        ) == {"kind": "retry"}
        assert session.events[-1].type == "llm/retry-started"

        # Already-aborted signal returns None cleanly without appending
        pre_aborted = AbortSignal()
        pre_aborted.abort("already cancelled")
        before_abort = len(session.events)
        assert (
            await _dispatch_request_error(
                ctx, session=session, retry_policy=policy, signal=pre_aborted
            )
            is None
        )
        assert len(session.events) == before_abort

        # Missing signal (None) and direct session payload still retry cleanly
        direct_payload: dict[str, Any] = {
            "session": session,
            "turn": 1,
            "step": 1,
            "provider": "mock",
            "failure": _make_failure(),
            "retryPolicy": policy,
            "signal": None,
        }
        assert await ctx.waterfall(
            "agent/request-error",
            direct_payload,
            default_handler=lambda _p: None,
        ) == {"kind": "retry"}

        # Signal aborted synchronously during llm/retry event commit prevents sleep
        reentrant_signal = AbortSignal()
        original_append = session.append

        def aborting_append(
            event_type: str, data: dict[str, Any], **kwargs: Any
        ) -> Any:
            ev = original_append(event_type, data, **kwargs)
            if event_type == "llm/retry":
                reentrant_signal.abort("reentrant cancel")
            return ev

        session.append = aborting_append  # type: ignore[method-assign]
        try:
            assert (
                await _dispatch_request_error(
                    ctx,
                    session=session,
                    retry_policy=policy,
                    signal=reentrant_signal,
                )
                is None
            )
        finally:
            session.append = original_append  # type: ignore[method-assign]

        # Aborting mid-sleep suppresses llm/retry-started
        long_policy = resolve_retry_policy(
            {
                "mode": "normal",
                "maxRetries": 5,
                "backoff": {
                    "initialDelayMs": 5000,
                    "maxDelayMs": 5000,
                    "jitterRatio": 0,
                },
            }
        )
        mid_signal = AbortSignal()
        task = asyncio.create_task(
            _dispatch_request_error(
                ctx, session=session, retry_policy=long_policy, signal=mid_signal
            )
        )
        await asyncio.sleep(0.02)
        mid_signal.abort("user cancel")
        assert await task is None
        assert session.events[-1].type == "llm/retry"

        # Custom sleep function racing against abort during always downstream settle
        always_policy = resolve_retry_policy(
            {
                "mode": "always",
                "backoff": {"initialDelayMs": 100, "maxDelayMs": 100, "jitterRatio": 0},
            }
        )
        down_signal = AbortSignal()

        async def aborting_downstream(_payload: dict[str, Any], _next: Any) -> Any:
            down_signal.abort("aborted downstream")
            return {"kind": "ignore"}

        dispose_down = ctx.on(
            "agent/request-error", aborting_downstream, mode=EventMode.WATERFALL
        )
        assert (
            await _dispatch_request_error(
                ctx, session=session, retry_policy=always_policy, signal=down_signal
            )
            is None
        )
        dispose_down()
    finally:
        registry.dispose()
        await ctx.fiber.dispose()


async def test_stale_callback_after_disposal_and_sleep_draining() -> None:
    """Disposing the plugin drains waits; a disposed retry listener is skipped.

    With the event bus's removal contract, a listener disposed mid-dispatch is
    skipped at its turn (not invoked), so the waterfall continues to the next
    live listener. The retry callback must not run; downstream is reached.
    """
    ctx, registry, session = _setup_context()
    disposed_sleep_called = asyncio.Event()

    async def slow_sleep(_delay_ms: float, _signal: AbortSignal) -> None:
        disposed_sleep_called.set()
        await asyncio.sleep(10)

    fiber = await _mount_retry(
        ctx, internals={"sleep": slow_sleep, "random": lambda: 0.5}
    )
    policy = resolve_retry_policy(
        {
            "mode": "normal",
            "maxRetries": 2,
            "backoff": {"initialDelayMs": 1000, "maxDelayMs": 1000, "jitterRatio": 0},
        }
    )

    in_flight = asyncio.create_task(
        _dispatch_request_error(ctx, session=session, retry_policy=policy)
    )
    await disposed_sleep_called.wait()
    await fiber.dispose()
    assert await in_flight is None
    registry.dispose()
    await ctx.fiber.dispose()

    # Upstream waterfall captures retry's callback before disposal, then invokes next() after disposal
    ctx2, registry2, session2 = _setup_context()
    fiber2 = await _mount_retry(ctx2, internals={"random": lambda: 0.5})

    downstream_calls = 0

    async def downstream_listener(
        _payload: Any, next_fn: Callable[[], Awaitable[Any]]
    ) -> Any:
        nonlocal downstream_calls
        downstream_calls += 1
        return await next_fn()

    dispose_downstream = ctx2.on(
        "agent/request-error", downstream_listener, mode=EventMode.WATERFALL
    )

    captured = asyncio.Event()
    release_gate = asyncio.Event()

    async def hold_before_retry(
        _payload: Any, next_fn: Callable[[], Awaitable[Any]]
    ) -> Any:
        captured.set()
        await release_gate.wait()
        return await next_fn()

    dispose_capture = ctx2.on(
        "agent/request-error",
        hold_before_retry,
        mode=EventMode.WATERFALL,
        prepend=True,
    )
    stale_task = asyncio.create_task(
        _dispatch_request_error(ctx2, session=session2, retry_policy=policy)
    )
    await captured.wait()
    await fiber2.dispose()
    release_gate.set()
    assert await stale_task is None
    # The disposed retry listener is skipped (no retry); downstream proceeds.
    assert downstream_calls == 1

    dispose_capture()
    dispose_downstream()
    registry2.dispose()
    await ctx2.fiber.dispose()


@pytest.mark.parametrize("outcome", ["ready", "cancelled", "disposed", "failed"])
async def test_overlapping_registration_outcomes(outcome: str) -> None:
    """Recovery overlapping deferred projection registration handles every lifecycle exit."""
    ctx, registry, session = _setup_context()
    original_register = registry.register
    registered_evt = asyncio.Event()
    release_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    async def deferred_register(
        definition: ProjectionDefinition[Any],
    ) -> ProjectionHandle[Any]:
        handle = original_register(definition)
        if definition.key == "llmRetry":
            registered_evt.set()
            await release_future
        return handle

    registry.register = deferred_register  # type: ignore[method-assign]
    waits: list[float] = []

    async def fast_sleep(delay_ms: float, _signal: AbortSignal) -> None:
        waits.append(delay_ms)

    # Do not await the load here: registration is deliberately deferred and
    # would block it; the original suite relied on fire-and-forget mounting.
    fiber = _start_retry(ctx, internals={"sleep": fast_sleep})
    await registered_evt.wait()

    signal = AbortSignal()
    recovering_evt = asyncio.Event()

    async def observe(_payload: dict[str, Any], next_fn: Any) -> Any:
        recovering_evt.set()
        return await next_fn()

    ctx.on("agent/request-error", observe, mode=EventMode.WATERFALL, prepend=True)

    policy = resolve_retry_policy(
        {
            "mode": "normal",
            "maxRetries": 1,
            "backoff": {"initialDelayMs": 1, "maxDelayMs": 1, "jitterRatio": 0},
        }
    )
    recover_task = asyncio.create_task(
        _dispatch_request_error(
            ctx, session=session, retry_policy=policy, signal=signal
        )
    )
    await recovering_evt.wait()

    if outcome == "cancelled":
        signal.abort("cancel during init")
    disposal_task = (
        asyncio.create_task(fiber.dispose()) if outcome == "disposed" else None
    )
    if disposal_task is not None:
        await asyncio.sleep(0)

    if outcome == "failed":
        release_future.set_exception(RuntimeError("retry registration failed"))
        with pytest.raises(RuntimeError, match="retry registration failed"):
            await fiber
        with pytest.raises(RuntimeError, match="retry registration failed"):
            await recover_task
    else:
        release_future.set_result(None)
        if disposal_task is not None:
            await disposal_task
        else:
            await fiber
        result = await recover_task
        if outcome == "ready":
            assert result == {"kind": "retry"}
            assert waits == [1]
        else:
            assert result is None
            assert waits == []

    retries = [e for e in session.events if e.type == "llm/retry"]
    assert len(retries) == (1 if outcome == "ready" else 0)
    registry.dispose()
    await ctx.fiber.dispose()


async def test_jsonl_persistence_roundtrip_and_loader_swapped_rows(
    tmp_path: Path,
) -> None:
    """Retry events survive JSONL persistence reload and llm-retry mounts in swapped order."""
    # Standalone llm-retry fiber stays pending until sessionProjections mounts.
    standalone_ctx = Context()
    pending_fiber = standalone_ctx.use(apply)
    await standalone_ctx.flush()
    assert pending_fiber.state != FiberState.ACTIVE
    await standalone_ctx.fiber.dispose()

    # Swapped row order: llm-retry BEFORE sessionProjections via load_profile.
    sessions_dir = tmp_path / "sessions"
    config_file = tmp_path / "swapped.yaml"
    rows = [
        {"id": "llm-retry", "name": "magitech_harness.llm.retry:apply", "config": {}},
        {
            "id": "session-projections",
            "name": "magitech_harness.session.projection_registry:SessionProjectionRegistry",
            "config": {},
        },
        {
            "id": "session-persistence-jsonl",
            "name": "magitech_harness.session.persistence_jsonl:JsonlSessionPersistenceService",
            "config": {"root": str(sessions_dir), "live_batch_delay": 0.2},
        },
        {
            "id": "sessions",
            "name": "magitech_harness.session.service:SessionService",
            "config": {"base_storage_dir": str(sessions_dir)},
        },
    ]
    with config_file.open("w", encoding="utf-8") as handle:
        dump_yaml(rows, handle)

    ctx = Context()
    load_profile(ctx, config_file)
    await ctx.flush()

    session = ctx.sessions.create("persist-retry")
    policy = resolve_retry_policy(
        {
            "mode": "normal",
            "maxRetries": 3,
            "backoff": {"initialDelayMs": 1, "maxDelayMs": 10, "jitterRatio": 0},
        }
    )
    assert await _dispatch_request_error(
        ctx, session=session, provider="mock", retry_policy=policy
    ) == {"kind": "retry"}

    await ctx.sessions.flush(session)
    saved_state = ctx.sessionProjections.snapshot(session)
    first_retry_id = session.events[0].payload["retryId"]
    await ctx.fiber.dispose()

    session_files = list(sessions_dir.rglob("*.jsonl"))
    assert len(session_files) == 1
    session_path = session_files[0]

    # Reload from disk into a fresh Session + fresh SessionProjectionRegistry
    ctx2 = Context()
    reg2 = SessionProjectionRegistry(ctx2)
    h2 = reg2.register(llm_retry_projection)
    reloaded = Session.load(session_path)
    assert math.isfinite(reloaded.events[-1].timestamp)
    assert h2.state_of(reloaded) == {
        retry_state_key("mock", retry_policy_key(policy)): {
            "retry": 1,
            "retryId": first_retry_id,
        }
    }
    assert saved_state is not None
    h2.dispose()
    reg2.dispose()
    await ctx2.fiber.dispose()


@pytest.mark.asyncio
async def test_disposed_retry_listener_returns_none_directly() -> None:
    """A retry listener invoked after its lifetime aborts short-circuits to None."""
    ctx, _registry, session = _setup_context()
    plugin = await _mount_retry(ctx, internals={"random": lambda: 0.5})
    listeners = ctx.events._listeners.get("agent/request-error", [])
    retry_listener = next(
        listener
        for listener in listeners
        if listener.callback.__name__ == "_on_request_error"
    )
    await plugin.dispose()
    payload = {
        "session": session,
        "provider": "p",
        "model": "m",
        "failure": {"message": "boom"},
    }
    result = await retry_listener.callback(payload, lambda *_a, **_k: None)
    assert result is None
