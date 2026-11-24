"""Provider-routed model-request retry policy on agent/request-error."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import random
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, NewType, TypedDict, cast

from animas import Context, EventMode

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.llm.retry_policy import ResolvedRetryPolicy
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    ProjectionHandle,
    SessionProjectionRegistry,
)
from magitech_harness.session.store import Session

name = "llm-retry"
inject = ["sessionProjections"]

RetryId = NewType("RetryId", str)


def retry_id(value: str) -> RetryId:
    """Wrap a retry chain identifier string in the ``RetryId`` brand."""
    return RetryId(value)


retryId = retry_id


class _NormalRetryEventData(TypedDict):
    """Durable payload recorded before a normal retry wait."""

    retryId: RetryId
    turn: int
    step: int
    provider: str
    mode: Literal["normal"]
    policyKey: str
    retry: int
    maxRetries: int
    delayMs: float
    failure: dict[str, Any]


class _AlwaysRetryEventData(TypedDict):
    """Durable payload recorded before an always-mode retry wait."""

    retryId: RetryId
    turn: int
    step: int
    provider: str
    mode: Literal["always"]
    policyKey: str
    retry: int
    delayMs: float
    failure: dict[str, Any]


LlmRetryEventData = _NormalRetryEventData | _AlwaysRetryEventData


class LlmRetryStartedEventData(TypedDict):
    """Durable transition recorded after one retry delay completes."""

    retryId: RetryId
    turn: int
    step: int
    retry: int


RequestErrorAction = dict[str, Any] | None
LlmRetryConfig = Mapping[str, Any]
RetryInternals = Mapping[str, Any]


def validate_config(config: Any) -> None:
    """Validate executor config, rejecting any key because providers own retryPolicy."""
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise TypeError("llm-retry: config must be a mapping")
    for key in config:
        if key == "retryPolicy":
            raise ValueError(
                "llm-retry: retryPolicy belongs under each provider configuration"
            )
        raise ValueError(f'llm-retry: unknown key "{key}"')


validateConfig = validate_config


def local_delay(
    config: ResolvedRetryPolicy,
    retry: int,
    random_fn: Callable[[], float] = random.random,
) -> float:
    """Compute bounded exponential backoff with symmetric jitter."""
    exponent = min(retry - 1, 1024)
    raw = (
        float(config["initialDelayMs"]) * math.ldexp(1.0, exponent)
        if exponent < 1024
        else math.inf
    )
    exponential = min(raw, float(config["maxDelayMs"]))
    jitter_ratio = float(config["jitterRatio"])
    jitter = 1.0 - jitter_ratio + 2.0 * jitter_ratio * float(random_fn())
    delay = min(exponential * jitter, float(config["maxDelayMs"]))
    if (
        jitter_ratio == 0.0
        and isinstance(config["initialDelayMs"], int)
        and isinstance(config["maxDelayMs"], int)
        and delay.is_integer()
    ):
        return int(delay)
    return delay


localDelay = local_delay


def retry_policy_key(policy: ResolvedRetryPolicy) -> str:
    """Build the canonical policy key string identifying a retry configuration."""
    if policy["mode"] == "always":
        return json.dumps(
            [
                policy["mode"],
                policy["initialDelayMs"],
                policy["maxDelayMs"],
                policy["jitterRatio"],
            ],
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return json.dumps(
        [
            policy["mode"],
            policy["maxRetries"],
            sorted(policy["retryableCodes"]),
            policy["initialDelayMs"],
            policy["maxDelayMs"],
            policy["jitterRatio"],
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )


retryPolicyKey = retry_policy_key


def retry_state_key(provider: str, policy_key: str) -> str:
    """Build the per-provider retry chain key stored in ``llmRetry`` projection."""
    return json.dumps([provider, policy_key], separators=(",", ":"), ensure_ascii=False)


retryStateKey = retry_state_key


def _validate_llm_retry_state(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError("llmRetry state must be a mapping")  # noqa: TRY004
    validated: dict[str, dict[str, Any]] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not isinstance(entry, Mapping):
            raise ValueError(  # noqa: TRY004
                "llmRetry state entries must be mappings keyed by strings"
            )
        if set(entry.keys()) != {"retry", "retryId"}:
            raise ValueError(
                "llmRetry state entry must have exact keys retry and retryId"
            )
        retry_val = entry["retry"]
        retry_id_val = entry["retryId"]
        if (
            isinstance(retry_val, bool)
            or not isinstance(retry_val, int)
            or retry_val < 0
        ):
            raise ValueError(
                "llmRetry state entry.retry must be a non-negative integer"
            )
        if not isinstance(retry_id_val, str):
            raise ValueError(  # noqa: TRY004
                "llmRetry state entry.retryId must be a string"
            )
        validated[key] = {"retry": retry_val, "retryId": RetryId(retry_id_val)}
    return validated


def _init_llm_retry_state(
    _header: dict[str, Any] | None = None, _inherited_count: int = 0
) -> dict[str, dict[str, Any]]:
    return {}


def _apply_llm_retry(
    state: dict[str, dict[str, Any]], event: SessionEvent
) -> dict[str, dict[str, Any]]:
    if event.type in ("step/start", "turn/end"):
        return {}
    if event.type != "llm/retry":
        return state
    data = event.payload
    key = retry_state_key(data["provider"], data["policyKey"])
    entry = state.get(key)
    if (
        entry is not None
        and entry["retry"] == data["retry"]
        and entry["retryId"] == data["retryId"]
    ):
        return state
    return {
        **state,
        key: {"retry": data["retry"], "retryId": RetryId(data["retryId"])},
    }


llm_retry_projection: ProjectionDefinition[dict[str, dict[str, Any]]] = (
    ProjectionDefinition(
        key="llmRetry",
        state_schema=_validate_llm_retry_state,
        init=_init_llm_retry_state,
        apply=_apply_llm_retry,
        wire=None,
        state_version=1,
    )
)
llmRetryProjection = llm_retry_projection


async def _cancellable_delay(
    delay_ms: float,
    signal: AbortSignal | None,
    lifetime: AbortSignal,
    sleep_fn: Callable[[float, AbortSignal], Awaitable[Any]] | None,
) -> bool:
    """Wait ``delay_ms`` milliseconds unless ``signal`` or ``lifetime`` aborts."""
    fused = AbortSignal()
    unsubs: list[Callable[[], None]] = []
    if signal is not None:
        unsubs.append(signal.on_abort(lambda: fused.abort(signal.reason)))
    unsubs.append(lifetime.on_abort(lambda: fused.abort(lifetime.reason)))
    try:
        if fused.is_aborted:
            return False
        aborted_evt = asyncio.Event()
        unsubs.append(fused.on_abort(aborted_evt.set))
        if sleep_fn is None:
            try:
                await asyncio.wait_for(
                    aborted_evt.wait(), timeout=max(0.0, float(delay_ms)) / 1000.0
                )
            except TimeoutError:
                return not fused.is_aborted
            return False
        sleep_task = asyncio.create_task(_await_sleep(sleep_fn, delay_ms, fused))
        abort_task = asyncio.create_task(aborted_evt.wait())
        done, pending = await asyncio.wait(
            {sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if sleep_task in done:
            await sleep_task
        return not fused.is_aborted
    finally:
        for unsub in unsubs:
            unsub()


async def _await_sleep(
    sleep_fn: Callable[[float, AbortSignal], Awaitable[Any]],
    delay_ms: float,
    fused: AbortSignal,
) -> None:
    await sleep_fn(delay_ms, fused)


def _extract_session(payload: Mapping[str, Any]) -> Session:
    session = payload.get("session")
    if isinstance(session, Session):
        return session
    agent = payload.get("agent")
    agent_session = getattr(agent, "session", None)
    if isinstance(agent_session, Session):
        return agent_session
    raise ValueError("llm-retry: agent/request-error payload must include session")


async def _settle_downstream(
    next_fn: Callable[[], Awaitable[Any]],
) -> tuple[str, Any]:
    try:
        return ("decision", await next_fn())
    except Exception as err:  # noqa: BLE001 - downstream failure is logged & recovered
        return ("error", err)


async def apply(
    ctx: Context,
    config: LlmRetryConfig | None = None,
    internals: RetryInternals | None = None,
) -> None:
    """Install provider-routed request recovery on ``agent/request-error``."""
    validate_config(config)
    registry = cast(
        SessionProjectionRegistry | None,
        ctx.get("sessionProjections", None, strict=False),
    )
    if registry is None:
        raise RuntimeError("llm-retry requires the sessionProjections service")

    random_fn: Callable[[], float] = (
        internals["random"]
        if internals is not None and "random" in internals
        else random.random
    )
    sleep_fn: Callable[[float, AbortSignal], Awaitable[Any]] | None = (
        internals["sleep"] if internals is not None and "sleep" in internals else None
    )

    owning_fiber = ctx.fiber
    lifetime = AbortSignal()
    active: set[asyncio.Task[RequestErrorAction]] = set()
    projection_handle: ProjectionHandle[dict[str, dict[str, Any]]] | None = None

    def _stopped(signal: AbortSignal | None = None) -> bool:
        return (
            lifetime.is_aborted
            or owning_fiber.uid is None
            or (signal is not None and signal.is_aborted)
        )

    raw_registration = registry.register(llm_retry_projection)

    async def _await_handle() -> ProjectionHandle[dict[str, dict[str, Any]]]:
        nonlocal projection_handle
        handle = (
            await raw_registration
            if inspect.isawaitable(raw_registration)
            else raw_registration
        )
        projection_handle = handle
        if _stopped():
            handle.dispose()
        return handle

    projection_ready: asyncio.Task[ProjectionHandle[dict[str, dict[str, Any]]]] = (
        asyncio.create_task(_await_handle())
    )

    async def _backoff(
        session: Session,
        turn: int,
        step: int,
        failure: dict[str, Any],
        provider: str,
        policy: ResolvedRetryPolicy,
        policy_key: str,
        retry_num: int,
        rid: RetryId,
        delay_ms: float,
        signal: AbortSignal | None,
    ) -> RequestErrorAction:
        failure_snapshot = copy.deepcopy(failure)
        if policy["mode"] == "normal":
            event_data: dict[str, Any] = {
                "retryId": rid,
                "turn": turn,
                "step": step,
                "provider": provider,
                "mode": policy["mode"],
                "policyKey": policy_key,
                "retry": retry_num,
                "maxRetries": policy["maxRetries"],
                "delayMs": delay_ms,
                "failure": failure_snapshot,
            }
        else:
            event_data = {
                "retryId": rid,
                "turn": turn,
                "step": step,
                "provider": provider,
                "mode": policy["mode"],
                "policyKey": policy_key,
                "retry": retry_num,
                "delayMs": delay_ms,
                "failure": failure_snapshot,
            }
        session.append("llm/retry", event_data)
        if not await _cancellable_delay(delay_ms, signal, lifetime, sleep_fn):
            return None
        session.append(
            "llm/retry-started",
            {"retryId": rid, "turn": turn, "step": step, "retry": retry_num},
        )
        return {"kind": "retry"}

    async def _recover(
        payload: Mapping[str, Any],
        next_fn: Callable[[], Awaitable[Any]],
    ) -> RequestErrorAction:
        policy: ResolvedRetryPolicy | None = payload.get("retryPolicy")
        if policy is None:
            return cast(RequestErrorAction, await next_fn())

        handle = await projection_ready
        signal: AbortSignal | None = payload.get("signal")
        if _stopped(signal):
            return None

        provider: str = payload["provider"]
        failure: dict[str, Any] = payload["failure"]
        if policy["mode"] == "always":
            outcome_kind, outcome_val = await _settle_downstream(next_fn)
            if _stopped(signal):
                return None
            if outcome_kind == "error":
                ctx.logger.warning(
                    'llm-retry: provider "%s" always policy ignored a downstream '
                    "recovery failure: %s",
                    provider,
                    outcome_val,
                )
            elif (
                isinstance(outcome_val, Mapping) and outcome_val.get("kind") == "retry"
            ):
                return dict(outcome_val)
        elif failure.get("code") not in policy["retryableCodes"]:
            return cast(RequestErrorAction, await next_fn())

        session = _extract_session(payload)
        policy_key = retry_policy_key(policy)
        retry_state = handle.state_of(session)
        previous = retry_state.get(retry_state_key(provider, policy_key))
        previous_retry = previous["retry"] if previous is not None else 0
        if policy["mode"] == "normal" and previous_retry >= policy["maxRetries"]:
            return cast(RequestErrorAction, await next_fn())

        retry_num = previous_retry + 1
        rid = (
            RetryId(previous["retryId"])
            if previous is not None
            else retry_id(str(uuid.uuid4()))
        )
        retry_after = failure.get("providerRetryAfterMs")
        if (
            not isinstance(retry_after, bool)
            and isinstance(retry_after, (int, float))
            and math.isfinite(float(retry_after))
            and retry_after > 0
        ):
            if retry_after > policy["maxDelayMs"]:
                if policy["mode"] == "normal":
                    return cast(RequestErrorAction, await next_fn())
                delay_ms = local_delay(policy, retry_num, random_fn)
            else:
                delay_ms = retry_after
        else:
            delay_ms = local_delay(policy, retry_num, random_fn)

        return await _backoff(
            session,
            payload["turn"],
            payload["step"],
            failure,
            provider,
            policy,
            policy_key,
            retry_num,
            rid,
            delay_ms,
            signal,
        )

    def _on_request_error(
        payload: Mapping[str, Any], next_fn: Callable[[], Awaitable[Any]]
    ) -> Awaitable[RequestErrorAction]:
        if lifetime.is_aborted:
            fut: asyncio.Future[RequestErrorAction] = (
                asyncio.get_running_loop().create_future()
            )
            fut.set_result(None)
            return fut
        task = asyncio.create_task(_recover(payload, next_fn))
        active.add(task)
        task.add_done_callback(active.discard)
        return task

    dispose_listener = ctx.on(
        "agent/request-error", _on_request_error, mode=EventMode.WATERFALL
    )

    async def _cleanup() -> None:
        dispose_listener()
        lifetime.abort("llm-retry plugin disposed")
        await asyncio.gather(*list(active), return_exceptions=True)
        if projection_handle is not None:
            projection_handle.dispose()

    ctx.effect(lambda: _cleanup, "llm-retry: abort and drain active recovery")
    await projection_ready


apply.inject = inject  # type: ignore[attr-defined]
