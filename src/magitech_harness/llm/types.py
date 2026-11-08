"""LLM models and message data structures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from magitech_harness.session.events import ToolCallData


class UsageData(BaseModel):
    """Token consumption telemetry."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


class ModelResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallData] = Field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: dict[str, Any] = Field(default_factory=dict)
    usage: UsageData = Field(default_factory=UsageData)


class ModelChunk(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallData] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: UsageData = Field(default_factory=UsageData)
