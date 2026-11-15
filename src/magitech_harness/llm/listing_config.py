"""Configuration defaults and validation for model-listing cache timing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, TypedDict

from magitech_harness.agent.cancellation import MAX_TIMER_DELAY_MS
from magitech_harness.llm.values import is_non_negative_safe_integer

DEFAULT_MODELS_TIMEOUT_MS = 5_000
DEFAULT_MODELS_MAX_AGE_MS = 600_000
LISTING_TIMEOUT = "MODEL_LISTING_TIMEOUT"


class ModelListingOptions(TypedDict, total=False):
    """Plain listing timing options accepted by resolve_listing_timing."""

    modelsTimeoutMs: float
    modelsMaxAgeMs: int


class ListingTiming(TypedDict):
    """Validated listing timing for one configuration generation."""

    timeoutMs: float
    maxAgeMs: int


class ListingSettings(ListingTiming):
    """Listing endpoint URL combined with validated timing configuration."""

    url: str


def resolve_listing_timing(
    options: Mapping[str, Any] | None,
    owner: str,
) -> ListingTiming:
    """Validate listing cache timing bounds, failing loud with *owner* prefix."""
    raw = options or {}
    timeout_ms = raw.get("modelsTimeoutMs", DEFAULT_MODELS_TIMEOUT_MS)
    max_age_ms = raw.get("modelsMaxAgeMs", DEFAULT_MODELS_MAX_AGE_MS)
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, (int, float))
        or not math.isfinite(timeout_ms)
        or timeout_ms <= 0
        or timeout_ms > MAX_TIMER_DELAY_MS
    ):
        raise ValueError(
            f"{owner}: modelsTimeoutMs must be a positive finite number "
            f"no greater than {MAX_TIMER_DELAY_MS}"
        )
    if not is_non_negative_safe_integer(max_age_ms):
        raise ValueError(f"{owner}: modelsMaxAgeMs must be a non-negative safe integer")
    return {
        "timeoutMs": timeout_ms,
        "maxAgeMs": max_age_ms,
    }


resolveListingTiming = resolve_listing_timing

__all__ = [
    "DEFAULT_MODELS_MAX_AGE_MS",
    "DEFAULT_MODELS_TIMEOUT_MS",
    "LISTING_TIMEOUT",
    "ListingSettings",
    "ListingTiming",
    "ModelListingOptions",
    "resolveListingTiming",
    "resolve_listing_timing",
]
