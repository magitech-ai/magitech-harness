"""Provider-owned request-retry policy validation and defaults."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from magitech_harness.llm.error import EMPTY_RESPONSE_CODE
from magitech_harness.llm.values import deep_freeze, is_safe_integer

MAX_TIMER_DELAY_MS = 2_147_483_647

DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_DELAY_MS = 500
DEFAULT_MAX_DELAY_MS = 10_000
DEFAULT_JITTER_RATIO = 0.1
DEFAULT_RETRYABLE_CODES: tuple[str, ...] = (
    EMPTY_RESPONSE_CODE,
    "RATE_LIMIT",
    "SERVER",
    "TIMEOUT",
    "TRANSPORT",
)

_NORMAL_POLICY_KEYS = frozenset({"mode", "maxRetries", "retryableCodes", "backoff"})
_ALWAYS_POLICY_KEYS = frozenset({"mode", "maxRetries", "retryableCodes", "backoff"})
_BACKOFF_KEYS = frozenset({"initialDelayMs", "maxDelayMs", "jitterRatio"})


class BackoffConfig(TypedDict, total=False):
    """Exponential backoff configuration accepted by provider retry policies."""

    initialDelayMs: float
    maxDelayMs: float
    jitterRatio: float


class NormalRetryPolicyConfig(TypedDict, total=False):
    """Bounded transient retry configuration."""

    mode: Literal["normal"]
    maxRetries: int
    retryableCodes: Sequence[str]
    backoff: BackoffConfig


class AlwaysRetryPolicyConfig(TypedDict, total=False):
    """Unbounded retry configuration for every model failure."""

    mode: Literal["always"]
    backoff: BackoffConfig


ResolvedRetryPolicy = dict[str, Any]

RetryPolicyConfig = (
    NormalRetryPolicyConfig | AlwaysRetryPolicyConfig | Mapping[str, Any]
)


def _validate_keys(
    mapping: Mapping[str, Any], allowed: frozenset[str], path: str
) -> None:
    for key in mapping:
        if key not in allowed:
            raise ValueError(f'{path}: unknown key "{key}"')


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _resolve_backoff(config: Any, path: str) -> dict[str, Any]:
    if config is not None:
        if not isinstance(config, Mapping):
            raise ValueError(f"{path} must be a mapping")
        _validate_keys(config, _BACKOFF_KEYS, path)
    backoff = config or {}
    initial_delay = (
        DEFAULT_INITIAL_DELAY_MS
        if backoff.get("initialDelayMs") is None
        else backoff["initialDelayMs"]
    )
    max_delay = (
        DEFAULT_MAX_DELAY_MS
        if backoff.get("maxDelayMs") is None
        else backoff["maxDelayMs"]
    )
    jitter_ratio = (
        DEFAULT_JITTER_RATIO
        if backoff.get("jitterRatio") is None
        else backoff["jitterRatio"]
    )

    if (
        not _is_finite_number(initial_delay)
        or initial_delay <= 0
        or initial_delay > MAX_TIMER_DELAY_MS
    ):
        raise ValueError(
            f"{path}.initialDelayMs must be a positive finite number no greater"
            f" than {MAX_TIMER_DELAY_MS}"
        )
    if (
        not _is_finite_number(max_delay)
        or max_delay <= 0
        or max_delay > MAX_TIMER_DELAY_MS
    ):
        raise ValueError(
            f"{path}.maxDelayMs must be a positive finite number no greater"
            f" than {MAX_TIMER_DELAY_MS}"
        )
    if initial_delay > max_delay:
        raise ValueError(
            f"{path}.initialDelayMs must be less than or equal to maxDelayMs"
        )
    if not _is_finite_number(jitter_ratio) or jitter_ratio < 0 or jitter_ratio > 1:
        raise ValueError(f"{path}.jitterRatio must be between 0 and 1")

    return {
        "initialDelayMs": initial_delay,
        "maxDelayMs": max_delay,
        "jitterRatio": jitter_ratio,
    }


def resolve_retry_policy(
    config: RetryPolicyConfig | None = None,
    path: str = "retryPolicy",
) -> dict[str, Any]:
    """Validate, default, and freeze a provider-owned retry policy."""
    if config is None:
        return deep_freeze(
            {
                "mode": "normal",
                "maxRetries": DEFAULT_MAX_RETRIES,
                "retryableCodes": list(DEFAULT_RETRYABLE_CODES),
                **_resolve_backoff(None, f"{path}.backoff"),
            }
        )
    if not isinstance(config, Mapping):
        raise TypeError(f"{path} must be a mapping")

    mode = config.get("mode")
    if mode == "normal":
        _validate_keys(config, _NORMAL_POLICY_KEYS, path)
        raw_max_retries = config.get("maxRetries")
        max_retries = (
            DEFAULT_MAX_RETRIES if raw_max_retries is None else raw_max_retries
        )
        raw_codes = config.get("retryableCodes")
        if raw_codes is None:
            retryable_codes = list(DEFAULT_RETRYABLE_CODES)
        elif isinstance(raw_codes, Sequence) and not isinstance(
            raw_codes, (str, bytes)
        ):
            retryable_codes = list(raw_codes)
        else:
            raise TypeError(f"{path}.retryableCodes must be a sequence of strings")
        if not is_safe_integer(max_retries) or max_retries < 0:
            raise ValueError(f"{path}.maxRetries must be a non-negative safe integer")
        if len(retryable_codes) == 0:
            raise ValueError(f"{path}.retryableCodes must not be empty")
        if any(not isinstance(code, str) or not code for code in retryable_codes):
            raise ValueError(
                f"{path}.retryableCodes must contain only non-empty strings"
            )
        if len(set(retryable_codes)) != len(retryable_codes):
            raise ValueError(f"{path}.retryableCodes must not contain duplicates")
        return deep_freeze(
            {
                "mode": "normal",
                "maxRetries": max_retries,
                "retryableCodes": retryable_codes,
                **_resolve_backoff(config.get("backoff"), f"{path}.backoff"),
            }
        )

    if mode == "always":
        _validate_keys(config, _ALWAYS_POLICY_KEYS, path)
        return deep_freeze(
            {
                "mode": "always",
                **_resolve_backoff(config.get("backoff"), f"{path}.backoff"),
            }
        )

    raise ValueError(f'{path}.mode must be "normal" or "always"')


resolveRetryPolicy = resolve_retry_policy
