"""Durable session event models for Magitech Harness.
Principle: Model-visible ⟺ Logged.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    TURN_START = "turn/start"
    TURN_END = "turn/end"
    STEP_START = "step/start"
    STEP_END = "step/end"
    USER_MESSAGE = "user/message"
    ASSISTANT_MESSAGE = "assistant/message"
    TOOL_CALL = "tool/call"
    TOOL_RESULT = "tool/result"
    COMPACTION = "session/compaction"


class ToolCallData(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResultData(BaseModel):
    call_id: str
    name: str
    content: Any
    is_error: bool = False


class SessionEvent(BaseModel):
    id: str
    session_id: str
    type: EventType
    timestamp: float = Field(default_factory=time.time)
    payload: dict[str, Any] = Field(default_factory=dict)
    ignorable: bool = False
    schema_version: int = 1
    seq: int = 0
