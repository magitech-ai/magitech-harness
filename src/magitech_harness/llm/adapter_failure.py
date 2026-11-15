"""Adapter failure normalization, LlmError, and credential usability checks."""

from __future__ import annotations

import math
from typing import Any

from magitech_harness.llm.api_key import normalize_api_key
from magitech_harness.llm.error import INVALID_CREDENTIAL_CODE, HarnessError
from magitech_harness.llm.types import LlmFailure
from magitech_harness.llm.values import FrozenDict, deep_freeze, is_safe_integer


def _validate_status(status: int) -> None:
    if (
        isinstance(status, bool)
        or not isinstance(status, int)
        or status < 100
        or status > 599
    ):
        raise ValueError("LlmError status must be an integer from 100 through 599")


def _validate_retry_after(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            "LlmError providerRetryAfterMs must be a positive finite number"
        )


def _validate_request_id(request_id: str) -> None:
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("LlmError requestId must be a non-empty string")


def _validate_offload_images(count: int) -> None:
    if not is_safe_integer(count) or count <= 0:
        raise ValueError("LlmError offloadImages must be a positive safe integer")


class LlmError(HarnessError):
    """Typed failure for LLM runtime and provider adapter operations."""

    failure: LlmFailure

    def __init__(
        self,
        message: str,
        code: str,
        *,
        status: int | None = None,
        provider_retry_after_ms: float | None = None,
        providerRetryAfterMs: float | None = None,
        request_id: str | None = None,
        requestId: str | None = None,
        offload_images: int | None = None,
        offloadImages: int | None = None,
        cause: BaseException | object | None = None,
    ) -> None:
        if not isinstance(message, str) or not message:
            raise ValueError("LlmError message must be a non-empty string")
        if not isinstance(code, str) or not code:
            raise ValueError("LlmError code must be a non-empty string")
        retry_after = (
            providerRetryAfterMs
            if providerRetryAfterMs is not None
            else provider_retry_after_ms
        )
        req_id = requestId if requestId is not None else request_id
        offload = offloadImages if offloadImages is not None else offload_images
        if status is not None:
            _validate_status(status)
        if retry_after is not None:
            _validate_retry_after(retry_after)
        if req_id is not None:
            _validate_request_id(req_id)
        if offload is not None:
            _validate_offload_images(offload)
        super().__init__(message, code, cause=cause)
        payload: dict[str, Any] = {"message": message, "code": code}
        if status is not None:
            payload["status"] = status
        if retry_after is not None:
            payload["providerRetryAfterMs"] = retry_after
        if req_id is not None:
            payload["requestId"] = req_id
        if offload is not None:
            payload["offloadImages"] = offload
        self.failure = deep_freeze(payload)  # type: ignore[assignment]


def assert_usable_api_key(raw: str, pkg: str, ref: str) -> str:
    """Validate and trim a supplied credential or raise INVALID_CREDENTIAL LlmError."""
    checked = normalize_api_key(raw)
    if checked["ok"]:
        return checked["value"]
    if checked["reason"] == "empty":
        raise LlmError(
            f"{pkg}: the API key resolved from {ref} is blank; set {ref} to the raw key"
            " (the web Models page writes it) or export it in the launching"
            " environment",
            INVALID_CREDENTIAL_CODE,
        )
    raise LlmError(
        f"{pkg}: the API key resolved from {ref} contains characters no HTTP header"
        f" can carry; set {ref} to the raw key alone (the web Models page writes it)",
        INVALID_CREDENTIAL_CODE,
    )


def _own_data_dict(err: BaseException) -> dict[str, Any]:
    """Read an exception's own instance dictionary without invoking class properties."""
    try:
        own = object.__getattribute__(err, "__dict__")
    except Exception:  # noqa: BLE001
        # Hostile __dict__ descriptors must not escape failure normalization.
        own = None
    return own if isinstance(own, dict) else {}


def _failure_snapshot(candidate: Any) -> LlmFailure | None:
    """Validate and detach a structured failure dictionary."""
    if type(candidate) not in (dict, FrozenDict):
        return None
    message = dict.get(candidate, "message")
    code = dict.get(candidate, "code")
    if not isinstance(message, str) or not message:
        return None
    if not isinstance(code, str) or not code:
        return None
    status = dict.get(candidate, "status")
    retry_after = dict.get(candidate, "providerRetryAfterMs")
    request_id = dict.get(candidate, "requestId")
    offload_images = dict.get(candidate, "offloadImages")

    if status is not None and (
        isinstance(status, bool)
        or not isinstance(status, int)
        or status < 100
        or status > 599
    ):
        return None
    if retry_after is not None and (
        isinstance(retry_after, bool)
        or not isinstance(retry_after, (int, float))
        or not math.isfinite(retry_after)
        or retry_after <= 0
    ):
        return None
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        return None
    if offload_images is not None and (
        not is_safe_integer(offload_images) or offload_images <= 0
    ):
        return None

    payload: dict[str, Any] = {"message": message, "code": code}
    if status is not None:
        payload["status"] = status
    if retry_after is not None:
        payload["providerRetryAfterMs"] = retry_after
    if request_id is not None:
        payload["requestId"] = request_id
    if offload_images is not None:
        payload["offloadImages"] = offload_images
    return deep_freeze(payload)  # type: ignore[return-value]


def _error_message(err: BaseException) -> str:
    """Read an exception message safely without letting hostile getters escape."""
    own_msg = _own_data_dict(err).get("message")
    if isinstance(own_msg, str) and own_msg:
        return own_msg
    try:
        rendered = str(err)
    except Exception:  # noqa: BLE001
        # Hostile exception __str__ getters at the boundary collapse safely.
        return "LLM adapter failed"
    return rendered if rendered else "LLM adapter failed"


def normalize_llm_failure(value: Any) -> LlmFailure:
    """Normalize any thrown value at an adapter boundary into frozen LlmFailure."""
    if not isinstance(value, BaseException):
        try:
            rendered = str(value) if value is not None else "null"
        except Exception:  # noqa: BLE001
            # Hostile value __str__ getters collapse to a neutral summary.
            rendered = "LLM adapter failed"
        message = rendered if rendered else "LLM adapter failed"
        return deep_freeze({"message": message, "code": "UNKNOWN"})  # type: ignore[return-value]

    own = _own_data_dict(value)
    carried = _failure_snapshot(own.get("failure"))
    if carried is not None and carried["code"] == own.get("code"):
        return carried

    code = value.code if isinstance(value, HarnessError) else "UNKNOWN"
    return deep_freeze({"message": _error_message(value), "code": code})  # type: ignore[return-value]


assertUsableApiKey = assert_usable_api_key
normalizeLlmFailure = normalize_llm_failure
