"""Message value helpers, context-summary bounds, and immutable factories."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from magitech_harness.llm.values import deep_freeze

CONTEXT_SUMMARY_MAX_CHARS = 120


def bound_context_summary(summary: str) -> str:
    """Ellipsize a notice summary if it exceeds CONTEXT_SUMMARY_MAX_CHARS."""
    if len(summary) <= CONTEXT_SUMMARY_MAX_CHARS:
        return summary
    return f"{summary[: CONTEXT_SUMMARY_MAX_CHARS - 1]}\u2026"


def freeze_message[T: Mapping[str, Any]](message: T) -> T:
    """Detach and deep-freeze an existing identified message without altering its id."""
    detached = copy.deepcopy(dict(message) if isinstance(message, Mapping) else message)
    return deep_freeze(detached)  # type: ignore[return-value]


def create_message(
    data: Mapping[str, Any] | None = None,
    *,
    role: str | None = None,
    content: Sequence[Mapping[str, Any]] | None = None,
    source: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create one identified, detached, deep-frozen conversation message."""
    payload = dict(data) if data is not None else {}
    if role is not None:
        payload["role"] = role
    if content is not None:
        payload["content"] = content
    if source is not None:
        payload["source"] = source
    payload.update(extra)
    detached = copy.deepcopy(payload)
    detached["id"] = str(uuid.uuid4())
    return deep_freeze(detached)


def create_developer_message(
    data: Mapping[str, Any] | None = None,
    *,
    content: Sequence[Mapping[str, Any]] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an identified, immutable developer message."""
    payload = dict(data) if data is not None else {}
    if content is not None:
        payload["content"] = content
    if source is not None:
        payload["source"] = source
    return create_message({**payload, "role": "developer"})


def create_user_message(
    data: Mapping[str, Any] | None = None,
    *,
    content: Sequence[Mapping[str, Any]] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an identified, immutable user message."""
    payload = dict(data) if data is not None else {}
    if content is not None:
        payload["content"] = content
    if source is not None:
        payload["source"] = source
    return create_message({**payload, "role": "user"})


def create_assistant_message(
    data: Mapping[str, Any] | None = None,
    *,
    content: Sequence[Mapping[str, Any]] | None = None,
    source: Mapping[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Create an identified, immutable assistant message with model source attribution."""
    payload = dict(data) if data is not None else {}
    resolved_content = content if content is not None else payload["content"]
    raw_source = dict(source if source is not None else payload.get("source") or {})
    if provider is not None:
        raw_source["provider"] = provider
    if model is not None:
        raw_source["model"] = model
    return create_message(
        {
            "role": "assistant",
            "content": resolved_content,
            "source": {"kind": "model", **raw_source},
        }
    )


def create_system_message(text: str) -> dict[str, Any]:
    """Create an identified, immutable system-prompt message."""
    return create_message(
        {
            "role": "system",
            "content": [] if not text else [{"type": "text", "text": text}],
            "source": {"kind": "system-prompt"},
        }
    )


def create_tool_result_message(
    data: Mapping[str, Any] | None = None,
    *,
    call_id: str | None = None,
    content: Sequence[Mapping[str, Any]] | None = None,
    is_error: bool | None = None,
) -> dict[str, Any]:
    """Create an identified, immutable tool-result message coupled to call_id."""
    payload = dict(data) if data is not None else {}
    resolved_call_id = call_id if call_id is not None else payload["callId"]
    resolved_content = content if content is not None else payload["content"]
    resolved_error = bool(is_error if is_error is not None else payload["isError"])
    return create_message(
        {
            "role": "tool",
            "source": {"kind": "tool", "callId": resolved_call_id},
            "toolCallId": resolved_call_id,
            "content": resolved_content,
            "isError": resolved_error,
        }
    )


boundContextSummary = bound_context_summary
freezeMessage = freeze_message
createMessage = create_message
createDeveloperMessage = create_developer_message
createUserMessage = create_user_message
createAssistantMessage = create_assistant_message
createSystemMessage = create_system_message
createToolResultMessage = create_tool_result_message
