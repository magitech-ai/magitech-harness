"""Unit tests for LlmError validation and adapter failure normalization."""

from __future__ import annotations

from typing import Any

import pytest

from magitech_harness.llm.adapter_failure import LlmError, normalize_llm_failure
from magitech_harness.llm.brand import ProviderRequestId
from magitech_harness.llm.error import HarnessError


def test_llm_error_validation_and_frozen_failure_snapshot() -> None:
    err = LlmError(
        "rate limited",
        "RATE_LIMIT",
        status=429,
        providerRetryAfterMs=1500.5,
        requestId=ProviderRequestId("req-1"),
        offloadImages=2,
        cause="upstream",
    )
    assert err.failure == {
        "message": "rate limited",
        "code": "RATE_LIMIT",
        "status": 429,
        "providerRetryAfterMs": 1500.5,
        "requestId": "req-1",
        "offloadImages": 2,
    }
    with pytest.raises(TypeError):
        err.failure["code"] = "OTHER"  # type: ignore[index]

    with pytest.raises(ValueError, match="non-empty string"):
        LlmError("", "RATE_LIMIT")
    with pytest.raises(ValueError, match="non-empty string"):
        LlmError("msg", "")
    for bad_status in (True, 99, 600, 429.5):
        with pytest.raises(ValueError, match="status must be an integer"):
            LlmError("msg", "ERR", status=bad_status)  # type: ignore[arg-type]
    for bad_delay in (False, 0, -5, float("inf")):
        with pytest.raises(ValueError, match="positive finite number"):
            LlmError("msg", "ERR", providerRetryAfterMs=bad_delay)
    with pytest.raises(ValueError, match="requestId must be a non-empty string"):
        LlmError("msg", "ERR", requestId=ProviderRequestId(""))
    for bad_offload in (True, 0, -1):
        with pytest.raises(ValueError, match="offloadImages"):
            LlmError("msg", "ERR", offloadImages=bad_offload)


def test_normalize_llm_failure_boundaries_and_hostiles() -> None:
    class _HostileStr:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert normalize_llm_failure(_HostileStr()) == {
        "message": "LLM adapter failed",
        "code": "UNKNOWN",
    }
    assert normalize_llm_failure("") == {
        "message": "LLM adapter failed",
        "code": "UNKNOWN",
    }
    assert normalize_llm_failure(None) == {"message": "null", "code": "UNKNOWN"}

    # HarnessError without custom snapshot keeps code.
    assert normalize_llm_failure(HarnessError("bad gateway", "SERVER")) == {
        "message": "bad gateway",
        "code": "SERVER",
    }

    # LlmError carries validated snapshot cleanly.
    llm_err = LlmError(
        "limited", "RATE_LIMIT", status=429, requestId=ProviderRequestId("r1")
    )
    assert normalize_llm_failure(llm_err) == {
        "message": "limited",
        "code": "RATE_LIMIT",
        "status": 429,
        "requestId": "r1",
    }

    # Property-backed accessors must be ignored; only own instance dict counts.
    class _AccessorError(Exception):
        @property
        def failure(self) -> dict[str, str]:
            return {"message": "ignored", "code": "IGNORED"}

        @property
        def code(self) -> str:
            return "IGNORED"

    assert normalize_llm_failure(_AccessorError("provider failed")) == {
        "message": "provider failed",
        "code": "UNKNOWN",
    }

    # Slots exception has no own user attributes.
    class _SlotsError(Exception):
        __slots__ = ()

    assert normalize_llm_failure(_SlotsError("slotted failure")) == {
        "message": "slotted failure",
        "code": "UNKNOWN",
    }

    # Hostile overriding __dict__ descriptor must not escape normalization.
    class _ThrowingDictError(Exception):
        @property
        def __dict__(self) -> dict[str, Any]:
            raise RuntimeError("hostile dict")

    assert normalize_llm_failure(_ThrowingDictError("dict boom")) == {
        "message": "dict boom",
        "code": "UNKNOWN",
    }

    # Malformed snapshot or mismatched code collapses safely.
    bad_snap = RuntimeError("provider failed")
    bad_snap.code = "FOREIGN"  # type: ignore[attr-defined]
    bad_snap.failure = {
        "message": "provider failed",
        "code": "FOREIGN",
        "requestId": "",
    }  # type: ignore[attr-defined]
    assert normalize_llm_failure(bad_snap) == {
        "message": "provider failed",
        "code": "UNKNOWN",
    }

    # Test snapshot edge validations cleanly.
    class _DictSub(dict[str, str]):
        pass

    for invalid_failure in (
        {"message": "", "code": "X"},
        {"message": "m", "code": ""},
        {"message": "m", "code": "X", "status": 99},
        {"message": "m", "code": "X", "providerRetryAfterMs": -1},
        {"message": "m", "code": "X", "offloadImages": 0},
        _DictSub({"message": "m", "code": "X"}),
        None,
    ):
        err: Any = RuntimeError("provider failed")
        err.code = "X"
        err.failure = invalid_failure
        assert normalize_llm_failure(err) == {
            "message": "provider failed",
            "code": "UNKNOWN",
        }

    # Valid optional fields inside snapshot without requestId/status still materialize subset.
    partial_err: Any = RuntimeError("ok")
    partial_err.code = "TIMEOUT"
    partial_err.failure = {
        "message": "timed out",
        "code": "TIMEOUT",
        "providerRetryAfterMs": 200,
        "offloadImages": 1,
    }
    assert normalize_llm_failure(partial_err) == {
        "message": "timed out",
        "code": "TIMEOUT",
        "providerRetryAfterMs": 200,
        "offloadImages": 1,
    }

    # Hostile __str__ on BaseException without own message falls back.
    class _HostileExc(Exception):
        def __str__(self) -> str:
            raise RuntimeError("str failed")

    assert normalize_llm_failure(_HostileExc()) == {
        "message": "LLM adapter failed",
        "code": "UNKNOWN",
    }

    empty_exc = RuntimeError()
    assert normalize_llm_failure(empty_exc) == {
        "message": "LLM adapter failed",
        "code": "UNKNOWN",
    }

    own_msg_exc = RuntimeError("ignored")
    own_msg_exc.message = "explicit message"  # type: ignore[attr-defined]
    assert normalize_llm_failure(own_msg_exc) == {
        "message": "explicit message",
        "code": "UNKNOWN",
    }
