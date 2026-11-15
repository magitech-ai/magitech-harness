"""Unit tests for branded LLM string identifier helpers."""

from __future__ import annotations

from magitech_harness.llm.brand import (
    AttachmentId,
    LlmAttemptId,
    MessageId,
    ProviderRequestId,
    ReasoningEffortId,
    ToolCallId,
    brand_string,
    brandString,
)


def test_brand_string_and_newtype_identities() -> None:
    assert brand_string("msg-1") == "msg-1"
    assert brandString("req-1") == "req-1"
    assert MessageId("m1") == "m1"
    assert ToolCallId("tc1") == "tc1"
    assert ProviderRequestId("pr1") == "pr1"
    assert LlmAttemptId("att1") == "att1"
    assert ReasoningEffortId("high") == "high"
    assert AttachmentId("file1") == "file1"
