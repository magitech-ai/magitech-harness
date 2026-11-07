"""
LLM models and message data structures.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from magitech_harness.session.events import ToolCallData


class ModelResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallData] = Field(default_factory=list)
    finish_reason: str = "stop"
    raw_response: dict[str, Any] = Field(default_factory=dict)
