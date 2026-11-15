"""Unit tests for listing timing configuration and validation."""

from __future__ import annotations

import pytest

from magitech_harness.llm.listing_config import (
    DEFAULT_MODELS_MAX_AGE_MS,
    DEFAULT_MODELS_TIMEOUT_MS,
    LISTING_TIMEOUT,
    resolve_listing_timing,
)
from magitech_harness.llm.retry_policy import MAX_TIMER_DELAY_MS


def test_resolve_listing_timing_defaults_and_custom() -> None:
    """Defaults match 0.2.2-rc.1 and valid custom values round-trip cleanly."""
    assert DEFAULT_MODELS_TIMEOUT_MS == 5_000
    assert DEFAULT_MODELS_MAX_AGE_MS == 600_000
    assert LISTING_TIMEOUT == "MODEL_LISTING_TIMEOUT"

    defaults = resolve_listing_timing(None, "test-owner")
    assert defaults == {
        "timeoutMs": 5_000,
        "maxAgeMs": 600_000,
    }

    custom = resolve_listing_timing(
        {"modelsTimeoutMs": 1250, "modelsMaxAgeMs": 0},
        "test-owner",
    )
    assert custom == {
        "timeoutMs": 1250,
        "maxAgeMs": 0,
    }


def test_resolve_listing_timing_rejects_out_of_range_values() -> None:
    """Invalid timeout or max-age bounds fail loud with owner prefix."""
    for bad_timeout in (
        0,
        -5,
        True,
        "5",
        float("nan"),
        float("inf"),
        MAX_TIMER_DELAY_MS + 1,
    ):
        with pytest.raises(ValueError, match="test-owner: modelsTimeoutMs"):
            resolve_listing_timing({"modelsTimeoutMs": bad_timeout}, "test-owner")

    for bad_age in (-1, 1.5, True, "10", 2**53):
        with pytest.raises(ValueError, match="test-owner: modelsMaxAgeMs"):
            resolve_listing_timing({"modelsMaxAgeMs": bad_age}, "test-owner")
