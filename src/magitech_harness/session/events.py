"""Durable session event models for Magitech Harness.
Principle: Model-visible ⟺ Logged.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Every event type the store can carry.

    The nine Phase-1 turn/step/message types plus every format-admitted v4
    type (the format layer's known set): the migration's own validated
    output must load, so the store vocabulary covers it. Two member names
    are shortened spellings of their frozen wire values; the values are
    unchanged. Per-type payload and lifecycle validation is later work.
    """

    TURN_START = "turn/start"
    TURN_END = "turn/end"
    STEP_START = "step/start"
    STEP_END = "step/end"
    USER_MESSAGE = "user/message"
    ASSISTANT_MESSAGE = "assistant/message"
    TOOL_CALL = "tool/call"
    TOOL_RESULT = "tool/result"
    COMPACTION = "session/compaction"
    AGENT_PRESET_SELECTED = "agent-preset/selected"
    AGENT_INBOX_SPLICED = "agent/inbox/spliced"
    APPROVAL_ASKED = "approval/asked"
    APPROVAL_DECIDED = "approval/decided"
    APPROVAL_POLICY = "approval/policy"
    ASSISTANT_ATTEMPT = "assistant/attempt"
    COMMAND_DONE = "command/done"
    COMMAND_RUN = "command/run"
    COMPACTION_END = "compaction/end"
    COMPACTION_PRUNE = "compaction/prune"
    COMPACTION_START = "compaction/start"
    COMPACTION_SUMMARY = "compaction/summary"
    DELIVERABLES_PRESENTED = "deliverables/presented"
    DEVELOPER_MESSAGE = "developer/message"
    FEEDBACK_MESSAGE_DELETE = "feedback/message-delete"
    FEEDBACK_MESSAGE_PUT = "feedback/message-put"
    FEEDBACK_RECORD = "feedback/record"
    GOAL_CHANGE = "goal/change"
    HOOK_INVOKED = "hook/invoked"
    HOOK_RESULT = "hook/result"
    IMAGE_OFFLOAD = "image/offload"
    JOB_REGISTERED = "job/registered"
    JOB_SETTLED = "job/settled"
    LLM_RETRY = "llm/retry"
    LLM_RETRY_STARTED = "llm/retry-started"
    MODEL_SELECTION = "model/selection"
    PERMISSION_PRESET = "permission/preset"
    PLAN_MODE = "plan/mode"
    REQUEST_CONTEXT = "request/context"
    REQUEST_HEADER = "request/header"
    SANDBOX_MODE = "sandbox/mode"
    SCHEDULE_CHANGE = "schedule/change"
    SESSION_LOG_DELIVERY_ACCEPTED = "session-log-deepseek/delivery-accepted"
    SESSION_END_SEED = "session/end-seed"
    SESSION_TITLE = "session/title"
    SESSION_TITLE_LLM_REQUEST = "session/title-llm-request"
    SUBAGENT_CATALOG = "subagent/catalog"
    SUBAGENT_DESCRIPTOR = "subagent/descriptor"
    SUBAGENT_MODEL_SELECTION_POLICY = "subagent/model-selection-policy"
    SYSTEM_MESSAGE = "system/message"
    TEAM_MEMBER = "team/member"
    TEAM_MESSAGE_DELIVERED = "team/message/delivered"
    TEAM_MESSAGE_QUEUED = "team/message/queued"
    TEAM_TASK = "team/task"
    TODO_WRITE = "todo/write"
    TOOL_WORKFLOW_AGENT_END = "tool-workflow/agent-end"
    TOOL_WORKFLOW_AGENT_START = "tool-workflow/agent-start"
    TOOL_WORKFLOW_RUN_END = "tool-workflow/run-end"
    TOOL_WORKFLOW_RUN_START = "tool-workflow/run-start"
    TOOL_PTC_DISPATCH = "tool/ptc-dispatch"
    TOOL_PTC_DISPATCH_START = "tool/ptc-dispatch-start"
    WEB_SEARCH_LLM_REQUEST = "web/deepseek-search-llm-request"
    WORKING_DIRECTORY_CHANGE = "working-directory/change"
    WORKSPACE_CHANGES = "workspace/changes"


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
    seq: int = 0
