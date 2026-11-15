"""Nominal string brand constructors shared across the LLM boundary."""

from __future__ import annotations

from typing import NewType

MessageId = NewType("MessageId", str)
ToolCallId = NewType("ToolCallId", str)
ProviderRequestId = NewType("ProviderRequestId", str)
LlmAttemptId = NewType("LlmAttemptId", str)
ReasoningEffortId = NewType("ReasoningEffortId", str)
AttachmentId = NewType("AttachmentId", str)


def brand_string(value: str) -> str:
    """Identity brand helper preserving the underlying string value."""
    return value


brandString = brand_string
