"""Unit tests for provider retry policy resolution and validation."""

from __future__ import annotations

from typing import Any

import pytest

from magitech_harness.llm.retry_policy import (
    MAX_TIMER_DELAY_MS,
    resolve_retry_policy,
)
from magitech_harness.llm.values import is_frozen


def test_resolve_retry_policy_defaults_normal_and_always() -> None:
    policy = resolve_retry_policy(None, "provider.retryPolicy")
    assert policy == {
        "mode": "normal",
        "maxRetries": 5,
        "retryableCodes": [
            "EMPTY_RESPONSE",
            "RATE_LIMIT",
            "SERVER",
            "TIMEOUT",
            "TRANSPORT",
        ],
        "initialDelayMs": 500,
        "maxDelayMs": 10_000,
        "jitterRatio": 0.1,
    }
    assert is_frozen(policy)
    assert is_frozen(policy["retryableCodes"])

    codes = ["BUSY"]
    custom = resolve_retry_policy(
        {
            "mode": "normal",
            "maxRetries": 4,
            "retryableCodes": codes,
            "backoff": {"initialDelayMs": 25, "maxDelayMs": 100, "jitterRatio": 0},
        },
        "provider.retryPolicy",
    )
    codes.append("LATE")
    assert custom == {
        "mode": "normal",
        "maxRetries": 4,
        "retryableCodes": ["BUSY"],
        "initialDelayMs": 25,
        "maxDelayMs": 100,
        "jitterRatio": 0,
    }

    assert resolve_retry_policy({"mode": "always"}, "provider.retryPolicy") == {
        "mode": "always",
        "initialDelayMs": 500,
        "maxDelayMs": 10_000,
        "jitterRatio": 0.1,
    }

    # Normal-only fields retained when switching to always mode are ignored.
    assert resolve_retry_policy(
        {"mode": "always", "maxRetries": 5, "retryableCodes": ["SERVER"]},
        "provider.retryPolicy",
    ) == {
        "mode": "always",
        "initialDelayMs": 500,
        "maxDelayMs": 10_000,
        "jitterRatio": 0.1,
    }


@pytest.mark.parametrize(
    ("config", "pattern"),
    [
        ("not-a-mapping", "must be a mapping"),
        ({"mode": "normal", "backoff": "bad"}, "backoff must be a mapping"),
        ({"mode": "normal", "retryableCodes": "SERVER"}, "sequence of strings"),
        ({"mode": "normal", "maxRetries": -1}, "maxRetries"),
        ({"mode": "normal", "maxRetries": 1.5}, "maxRetries"),
        ({"mode": "always", "backoff": {"initialDelayMs": 0}}, "initialDelayMs"),
        ({"mode": "normal", "backoff": {"maxDelayMs": float("inf")}}, "maxDelayMs"),
        (
            {"mode": "normal", "backoff": {"initialDelayMs": MAX_TIMER_DELAY_MS + 1}},
            "initialDelayMs",
        ),
        (
            {"mode": "always", "backoff": {"maxDelayMs": MAX_TIMER_DELAY_MS + 1}},
            "maxDelayMs",
        ),
        (
            {"mode": "normal", "backoff": {"initialDelayMs": 20, "maxDelayMs": 10}},
            "less than or equal",
        ),
        ({"mode": "always", "backoff": {"jitterRatio": 1.1}}, "jitterRatio"),
        ({"mode": "always", "backoff": {"jitterRatio": -0.1}}, "jitterRatio"),
        ({"mode": "always", "backoff": {"jitterRatio": True}}, "jitterRatio"),
        ({"mode": "normal", "retryableCodes": []}, "must not be empty"),
        ({"mode": "normal", "retryableCodes": ["SERVER", "SERVER"]}, "duplicates"),
        ({"mode": "normal", "retryableCodes": [""]}, "non-empty strings"),
        ({"mode": "normal", "retryableCodes": [429]}, "non-empty strings"),
        ({"mode": "normal", "maxRetires": 1}, 'unknown key "maxRetires"'),
        (
            {"mode": "always", "backoff": {"initialDelay": 1}},
            'unknown key "initialDelay"',
        ),
        ({"mode": "sometimes"}, 'mode must be "normal" or "always"'),
    ],
)
def test_resolve_retry_policy_rejects_invalid_inputs(config: Any, pattern: str) -> None:
    with pytest.raises((TypeError, ValueError), match=pattern):
        resolve_retry_policy(config, "provider.retryPolicy")
