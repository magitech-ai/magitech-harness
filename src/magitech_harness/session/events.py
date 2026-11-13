"""Durable session event models for Magitech Harness.
Principle: Model-visible ⟺ Logged.
"""

from __future__ import annotations

import math
import re
import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Event vocabulary: the 62-type registry, reproduced as a merge.
#
# The core map declares 14 types; the other 48 arrive by declaration
# merging from their owner packages. The registry is built the same way:
# a core set plus an owner -> types mapping whose union is the known set.
# No 63rd type exists; the retired ``session/compaction`` spelling is not
# a known type and is refused as unknown unless an event carries it with
# ``ignorable: true``.
# ---------------------------------------------------------------------------

CORE_EVENT_TYPES: frozenset[str] = frozenset(
    [
        "assistant/attempt",
        "assistant/message",
        "developer/message",
        "request/context",
        "request/header",
        "session/end-seed",
        "step/end",
        "step/start",
        "system/message",
        "tool/call",
        "tool/result",
        "turn/end",
        "turn/start",
        "user/message",
    ]
)
"""The 14 event types declared by the session core map itself."""

OWNER_EVENT_TYPES: dict[str, tuple[str, ...]] = {
    "preset/agent-preset-registry": ("agent-preset/selected",),
    "core/agent": ("agent/inbox/spliced", "model/selection"),
    "interaction/user-approval": (
        "approval/asked",
        "approval/decided",
        "approval/policy",
    ),
    "interaction/commands": ("command/done", "command/run"),
    "compaction/compaction": (
        "compaction/end",
        "compaction/prune",
        "compaction/start",
        "compaction/summary",
    ),
    "deliverables/tool-present": ("deliverables/presented",),
    "feedback/message-feedback": (
        "feedback/message-delete",
        "feedback/message-put",
    ),
    "feedback/command-feedback": ("feedback/record",),
    "goal/goal": ("goal/change",),
    "experimental/hook-protocol": ("hook/invoked", "hook/result"),
    "compaction/compaction-image-offload": ("image/offload",),
    "jobs/jobs": ("job/registered", "job/settled"),
    "llm/llm-retry": ("llm/retry", "llm/retry-started"),
    "interaction/permission-presets": ("permission/preset",),
    "plan/plan-mode": ("plan/mode",),
    "sandbox/sandbox-policy": ("sandbox/mode",),
    "schedule/schedule": ("schedule/change",),
    "session/session-log": ("session-log-deepseek/delivery-accepted",),
    "session/session-title": ("session/title", "session/title-llm-request"),
    "subagent/subagent": ("subagent/catalog", "subagent/descriptor"),
    "subagent/tool-subagent": ("subagent/model-selection-policy",),
    "experimental/agent-team": (
        "team/member",
        "team/message/delivered",
        "team/message/queued",
        "team/task",
    ),
    "todo/tool-todo": ("todo/write",),
    "workflow/tool-workflow": (
        "tool-workflow/agent-end",
        "tool-workflow/agent-start",
        "tool-workflow/run-end",
        "tool-workflow/run-start",
    ),
    "core/tool-executor-ptc": ("tool/ptc-dispatch", "tool/ptc-dispatch-start"),
    "web/web-search": ("web/deepseek-search-llm-request",),
    "session/working-directory": ("working-directory/change",),
    "deliverables/workspace-changes": ("workspace/changes",),
}
"""Owner package -> the event types it merges into the vocabulary (48)."""

KNOWN_SESSION_EVENT_TYPES: frozenset[str] = frozenset(
    set(CORE_EVENT_TYPES)
    | {event_type for types in OWNER_EVENT_TYPES.values() for event_type in types}
)
"""Every event type this build understands at format v4 (62)."""

SURFACE_EVENT_TYPES: frozenset[str] = frozenset(
    [
        "system/message",
        "developer/message",
        "user/message",
        "assistant/message",
        "tool/result",
    ]
)
"""Event types that may carry a surface operation (5)."""

MESSAGE_PROJECTION_EVENT_TYPES: frozenset[str] = frozenset(["image/offload"])
"""Event types whose model-visible effects require a registered projection."""

_PLUGIN_RECORD_PATTERN = re.compile(
    r"^plugin:[a-z0-9][a-z0-9._-]*(/[a-z0-9][a-z0-9._-]*)*$"
)


def is_plugin_record_type(value: str) -> bool:
    """Whether a type follows the ``plugin:`` record grammar.

    Slash-separated segments of lowercase letters, digits, ``.``, ``_``,
    and ``-``, each starting with a letter or digit. A record type is
    never a member of the known event vocabulary.
    """
    return _PLUGIN_RECORD_PATTERN.match(value) is not None


class EventType(str, Enum):
    """Every known event type the store can carry (62)."""

    TURN_START = "turn/start"
    TURN_END = "turn/end"
    STEP_START = "step/start"
    STEP_END = "step/end"
    USER_MESSAGE = "user/message"
    ASSISTANT_MESSAGE = "assistant/message"
    TOOL_CALL = "tool/call"
    TOOL_RESULT = "tool/result"
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
    error: dict[str, Any] | None = None


class SessionEvent(BaseModel):
    """One committed entry in a session log.

    ``type`` is a plain string (not the enum) so an unknown event carried
    opaquely with ``ignorable: true`` — and a ``plugin:`` record — can be
    represented exactly as restored. Known types compare equal to their
    :class:`EventType` member because the enum is a ``str`` enum.
    """

    id: str
    session_id: str
    type: str
    timestamp: float = Field(default_factory=time.time)
    payload: dict[str, Any] = Field(default_factory=dict)
    ignorable: bool = False
    seq: int = 0
    surface_op: str | dict[str, Any] | None = None
    source_event_seqs: list[int] | None = None


class _Rejected:
    """Sentinel distinguishing a rejected JSON value from a JSON null."""


_REJECTED = _Rejected()


def _snapshot_json_value(value: Any, seen: set[int]) -> Any:
    """Deep-copy one value through the lossless-JSON domain.

    Returns :data:`_REJECTED` for anything the durable log cannot carry:
    non-finite numbers, negative zero, non-string dict keys, and values
    outside the JSON value domain (functions, symbols, dates, sets, class
    instances, circular references).
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or value == 0.0 and math.copysign(1.0, value) < 0:
            return _REJECTED
        return value
    if isinstance(value, list):
        if id(value) in seen:
            return _REJECTED
        seen.add(id(value))
        copied = [_snapshot_json_value(item, seen) for item in value]
        seen.discard(id(value))
        if any(item is _REJECTED for item in copied):
            return _REJECTED
        return copied
    if isinstance(value, dict):
        if id(value) in seen:
            return _REJECTED
        if any(not isinstance(key, str) for key in value):
            return _REJECTED
        seen.add(id(value))
        copied_dict = {
            key: _snapshot_json_value(item, seen) for key, item in value.items()
        }
        seen.discard(id(value))
        if any(item is _REJECTED for item in copied_dict.values()):
            return _REJECTED
        return copied_dict
    return _REJECTED


def snapshot_json_value(value: Any) -> Any:
    """Return a detached lossless-JSON copy of ``value``, or ``None``.

    A rejected (non-JSON-serializable) value returns ``None``; callers
    only pass values that are never a bare JSON null (event data and
    surface metadata are always objects here), so ``None`` unambiguously
    means rejection at those call sites.
    """
    result = _snapshot_json_value(value, set())
    return None if result is _REJECTED else result


def is_lossless_json(value: Any) -> bool:
    """Whether ``value`` round-trips through the lossless-JSON domain."""
    return _snapshot_json_value(value, set()) is not _REJECTED
