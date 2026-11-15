"""Harness error base, failure classifiers, cause rendering, and remote error."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, TypeGuard

CONTEXT_WINDOW_EXCEEDED_CODE = "CONTEXT_WINDOW_EXCEEDED"
QUOTA_EXCEEDED_CODE = "QUOTA"
ACCOUNT_QUOTA_EXCEEDED_CODE = "ACCOUNT_QUOTA"
EMPTY_RESPONSE_CODE = "EMPTY_RESPONSE"
INVALID_CREDENTIAL_CODE = "INVALID_CREDENTIAL"
IMAGE_OFFLOAD_REQUIRED_CODE = "IMAGE_OFFLOAD_REQUIRED"

_STRUCTURED_CONTEXT_OVERFLOW = re.compile(
    r"(?:^|[^a-z0-9])context[\s_-](?:length|window)[\s_-]"
    r"(?:exceed(?:ed|s)?|overflow(?:ed)?|limit[\s_-]exceeded)(?:$|[^a-z0-9])",
    re.IGNORECASE,
)
_MAX_CONTEXT_LENGTH = re.compile(
    r"\b(?:maximum|max)(?:\s+(?:allowed|supported))?\s+context\s+(?:length|window)\b",
    re.IGNORECASE,
)
_TOO_LARGE_FOR_CONTEXT = re.compile(
    r"\b(?:request|prompt|input|messages?)\s+(?:is\s+|are\s+)?"
    r"too\s+(?:large|long)\s+for\s+(?:(?:this|the)\s+)?"
    r"(?:model(?:'s)?\s+)?context(?:\s+window)?\b",
    re.IGNORECASE,
)
_TOO_LARGE_FOR_MODEL = re.compile(
    r"\b(?:input|prompt|request)\s+(?:is\s+)?too\s+(?:long|large)\s+"
    r"for\s+(?:this|the)\s+model\b",
    re.IGNORECASE,
)
_EXCEEDS_MODEL_CONTEXT = re.compile(
    r"\b(?:input|prompt|request|messages?)\b.{0,40}"
    r"\b(?:exceed(?:s|ed)?|overflows?|is\s+larger\s+than)\b.{0,40}"
    r"\b(?:the\s+)?(?:model(?:'s)?\s+)?context(?:\s+(?:length|window))?\b",
    re.IGNORECASE,
)

_INSUFFICIENT_QUOTA = re.compile(
    r"\binsufficient[\s_-]+(?:quota|balance|credits?)\b", re.IGNORECASE
)
_QUOTA_EXCEEDED = re.compile(
    r"\b(?:quota|usage[\s_-]+limit)[\s_-]+(?:exceeded|exhausted|reached)\b",
    re.IGNORECASE,
)
_EXCEEDED_QUOTA = re.compile(
    r"\bexceed(?:ed|s)?[\s_-]+(?:(?:your|the)[\s_-]+)?(?:current[\s_-]+)?quota\b",
    re.IGNORECASE,
)
_BALANCE_EXHAUSTED = re.compile(
    r"\b(?:balance|credits?)[\s_-]+(?:exhausted|depleted)\b", re.IGNORECASE
)
_OUT_OF_CREDITS = re.compile(
    r"\bout[\s_-]+of[\s_-]+(?:credits?|budget)\b", re.IGNORECASE
)


class HarnessError(Exception):
    """Base error carrying a stable machine-routable code and optional cause."""

    code: str

    def __init__(
        self,
        message: str,
        code: str,
        *,
        cause: BaseException | object | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.cause: Any = cause
        if isinstance(cause, BaseException):
            self.__cause__ = cause


class RemoteError(Exception):
    """Structured remote boundary failure with stable code and details mapping."""

    is_mth_remote_error: bool = True

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, Any],
        *,
        cause: BaseException | object | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details)
        self.cause: Any = cause
        if isinstance(cause, BaseException):
            self.__cause__ = cause


def is_harness_error(value: Any) -> TypeGuard[HarnessError]:
    """Return True when *value* is a :class:`HarnessError` instance."""
    return isinstance(value, HarnessError)


def is_context_window_exceeded_error(detail: str) -> bool:
    """Classify provider error text identifying context-window overflow."""
    return bool(
        _STRUCTURED_CONTEXT_OVERFLOW.search(detail)
        or _MAX_CONTEXT_LENGTH.search(detail)
        or _TOO_LARGE_FOR_CONTEXT.search(detail)
        or _TOO_LARGE_FOR_MODEL.search(detail)
        or _EXCEEDS_MODEL_CONTEXT.search(detail)
    )


def is_quota_exceeded_error(detail: str) -> bool:
    """Classify terminal quota, balance, credit, or usage-limit failures."""
    return bool(
        _INSUFFICIENT_QUOTA.search(detail)
        or _QUOTA_EXCEEDED.search(detail)
        or _EXCEEDED_QUOTA.search(detail)
        or _BALANCE_EXHAUSTED.search(detail)
        or _OUT_OF_CREDITS.search(detail)
    )


def _aggregate_members(err: BaseException) -> Sequence[Any] | None:
    """Extract sub-errors from ExceptionGroup or AggregateError-like objects."""
    if isinstance(err, BaseExceptionGroup):
        return err.exceptions
    members = getattr(err, "errors", None)
    if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
        return members
    return None


def _exception_message(err: BaseException) -> str:
    """Read an exception message defensively, falling back to its class name."""
    try:
        own_dict = object.__getattribute__(err, "__dict__")
    except Exception:  # noqa: BLE001
        # Hostile __dict__ descriptors fall back to str(err).
        own_dict = None
    own_msg = own_dict.get("message") if isinstance(own_dict, dict) else None
    if isinstance(own_msg, str):
        rendered = own_msg
    elif isinstance(err, BaseExceptionGroup):
        rendered = err.message
    else:
        rendered = str(err)
    return rendered if rendered else type(err).__name__


def error_chain(value: Any) -> str:
    """Render an arbitrary thrown value with its full cause chain and aggregates.

    Hostile attribute getters or string conversions collapse only their node to
    ``'<unrenderable value>'``. Active-path cycles render as
    ``'<circular cause>'``, while diamond-shared nodes render in full along each
    branch.
    """
    active_path: set[int] = set()

    def _render(current: Any) -> str:
        current_id = id(current)
        if current_id in active_path:
            return "<circular cause>"
        active_path.add(current_id)
        try:
            if not isinstance(current, BaseException):
                if isinstance(current, dict):
                    msg = current.get("message")
                    if isinstance(msg, str):
                        return msg
                return str(current)
            message = _exception_message(current)
            members_seq = _aggregate_members(current)
            if members_seq is not None and len(members_seq) > 0:
                rendered_members = "; ".join(_render(m) for m in members_seq)
                members_text = f" [{rendered_members}]"
            else:
                members_text = ""
            raw_cause = getattr(current, "cause", None)
            if raw_cause is None:
                raw_cause = current.__cause__
            cause_text = "" if raw_cause is None else _render(raw_cause)
            cause_suffix = (
                "" if cause_text == "" or cause_text == message else f": {cause_text}"
            )
            return f"{message}{members_text}{cause_suffix}"
        except Exception:  # noqa: BLE001
            # Hostile getters, custom __str__, or recursive errors collapse safely.
            return "<unrenderable value>"
        finally:
            active_path.discard(current_id)

    return _render(value)


isHarnessError = is_harness_error
isContextWindowExceededError = is_context_window_exceeded_error
isQuotaExceededError = is_quota_exceeded_error
errorChain = error_chain
