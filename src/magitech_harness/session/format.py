"""Durable session format: header-only classification, format v4, frozen migrations.

The session log is one JSON object per line. The first line is the physical
header record (``type: 'session'`` plus the logical header fields); every
following line is one physical event record (``type`` / ``seq`` / ``time`` /
``data`` plus the optional envelope members).

Format versions are a single monotonic integer; the current version is 4.
Adjacent versions are connected by frozen migrations (v0->v1->v2->v3->v4) and
nothing else: a stored log is either current, migratable through the chain,
or refused. Corrupt or foreign logs are refused, never silently repaired.

Storage is plain JSONL for Phase 2: no compression framing. The reference
implementation defaults to checksummed compressed frames, one per header or
batch; this port records the plain-JSONL call here deliberately (see the
module note in the project plan) and revisits it only with evidence of size
pressure. The only behavioral differences are the filename suffix, no
checksum protection, and no frame boundaries, so torn-tail detection rests
on the JSONL newline/record rules.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeGuard, cast

# ---------------------------------------------------------------------------
# Format version
# ---------------------------------------------------------------------------

CURRENT_FORMAT_VERSION = 4
"""The single monotonic format version this build reads and writes."""

MAX_SAFE_INTEGER = 2**53 - 1

JsonValue = Any
JsonObject = dict[str, Any]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SessionFormatError(Exception):
    """A durable session artifact is malformed and cannot be restored."""


class SessionFormatUnsupportedMigrationError(SessionFormatError):
    """A readable artifact has no supported migration in this build."""


class SessionFormatUnsupportedError(Exception):
    """The stored log is intact but this build cannot faithfully interpret it.

    Raised at the persistence seam (never by the format layer itself), for a
    header carrying an unsupported format version or an event type unknown
    to this build. Distinct from corruption: nothing is damaged, and the raw
    log remains readable at ``location`` when one exists.
    """

    def __init__(self, message: str, location: str | None = None) -> None:
        super().__init__(message)
        self.location = location


class SessionPersistenceCorruptionError(Exception):
    """Durable session contents failed validation after a successful read."""

    def __init__(self, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.__cause__ = cause


def session_format_version_refusal(session_id: str, version: int) -> str:
    """Direction-aware refusal text for a stored session this build cannot read.

    A newer version names the upgrade; an older version without an upgrade
    path says so. The text is stable: callers surface it verbatim.
    """
    if version > CURRENT_FORMAT_VERSION:
        return (
            f'session "{session_id}" uses log format v{version}, but this harness'
            f" reads only v{CURRENT_FORMAT_VERSION}: the log was written by a newer"
            " harness — upgrade the harness to open it"
        )
    return (
        f'session "{session_id}" uses log format v{version}, older than the'
        f" supported v{CURRENT_FORMAT_VERSION}, and this build ships no upgrade"
        " path for it"
    )


# ---------------------------------------------------------------------------
# JSON boundary helpers
# ---------------------------------------------------------------------------


def is_json_object(value: Any) -> TypeGuard[JsonObject]:
    """Whether the value is a non-null, non-array JSON object."""
    return isinstance(value, dict)


def _is_safe_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and -MAX_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER
    )


def require_count(value: Any, label: str) -> int:
    """Require a non-negative safe integer."""
    if not _is_safe_integer(value) or value < 0:
        raise SessionFormatError(f"{label} must be a non-negative safe integer")
    return cast(int, value)


def require_safe_integer(value: Any, label: str) -> int:
    """Require a safe integer."""
    if not _is_safe_integer(value):
        raise SessionFormatError(f"{label} must be a safe integer")
    return cast(int, value)


def require_format_version(value: Any, label: str = "Session format version") -> int:
    """Require a non-negative integral format version."""
    return require_count(value, label)


def require_record(value: Any, label: str) -> JsonObject:
    """Require a plain JSON object."""
    if not is_json_object(value):
        raise SessionFormatError(f"{label} must be a JSON object")
    return cast(JsonObject, value)


def exact_keys(
    record: JsonObject,
    required: list[str] | tuple[str, ...],
    optional: list[str] | tuple[str, ...],
    label: str,
) -> None:
    """Require every named member and no member outside the admitted lists."""
    allowed = set(required) | set(optional)
    for key in record:
        if key not in allowed:
            raise SessionFormatError(f"{label} has unexpected member {json.dumps(key)}")
    for key in required:
        if key not in record:
            raise SessionFormatError(f"{label} lacks required member {json.dumps(key)}")


def inspect_format_version(header_value: Any) -> int:
    """Read only the version field of a physical header value.

    Never inspects event rows. Raises :class:`SessionFormatError` when the
    value is not an object or carries no valid version.
    """
    if not is_json_object(header_value):
        raise SessionFormatError("Session header must be a JSON object")
    return require_format_version(header_value.get("version"))


def snapshot_header(header: JsonObject, label: str = "Session header") -> JsonObject:
    """Validate a logical header without inspecting an event body."""
    if not is_json_object(header):
        raise SessionFormatError(f"{label} must be a JSON object")
    inspect_format_version(header)
    if not isinstance(header.get("id"), str):
        raise SessionFormatError(f"{label} id must be a string")
    require_count(header.get("createdAt"), f"{label} createdAt")
    if not isinstance(header.get("isSeeded"), bool):
        raise SessionFormatError(f"{label} isSeeded must be a boolean")
    require_count(header.get("delegationDepth"), f"{label} delegationDepth")
    return header


# ---------------------------------------------------------------------------
# Generation filenames
# ---------------------------------------------------------------------------


def session_format_log_filename(version: int) -> str:
    """Canonical raw log basename for one format generation.

    Version zero keeps the original ``session.jsonl``; every later
    generation carries a lowercase numeric ``.vN`` component.
    """
    generation = require_format_version(version, "Session log generation version")
    if generation == 0:
        return "session.jsonl"
    return f"session.v{generation}.jsonl"


def parse_session_format_log_filename(filename: str) -> int | None:
    """Read the generation named by one raw JSONL log basename.

    Temporary, uppercase, leading-zero, ``.v0``, and compression-suffixed
    names are not canonical and return ``None``.
    """
    if filename == "session.jsonl":
        return 0
    if not filename.startswith("session.v") or not filename.endswith(".jsonl"):
        return None
    middle = filename[len("session.v") : -len(".jsonl")]
    if not middle or (len(middle) > 1 and middle.startswith("0")):
        return None
    if not middle.isdigit():
        return None
    version = int(middle)
    if version < 1 or version > MAX_SAFE_INTEGER:
        return None
    return version


# ---------------------------------------------------------------------------
# Seq-range encoding
# ---------------------------------------------------------------------------


def encode_seq_ranges(values: list[int]) -> list[Any]:
    """Replace profitable consecutive runs with inclusive ``[start, end]`` pairs.

    A run of three or more consecutive values becomes one pair; shorter runs
    stay as individual numbers. A non-increasing input is returned unchanged.
    """
    if any(values[i] <= values[i - 1] for i in range(1, len(values))):
        return list(values)
    encoded: list[Any] = []
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[end + 1] == values[end] + 1:
            end += 1
        if end - start >= 2:
            encoded.append([values[start], values[end]])
        else:
            for index in range(start, end + 1):
                encoded.append(values[index])
        start = end + 1
    return encoded


def decode_seq_ranges(value: Any, max_entries: int | None = None) -> list[int]:
    """Expand a stored ``sourceEventSeqs`` value into plain sequence numbers.

    Plain members and inclusive ``[start, end]`` pairs are admitted. When
    ``max_entries`` is given, every member must be below it. A stored range
    forces the whole output to be strictly increasing.
    """
    if not isinstance(value, list):
        raise SessionFormatError("sourceEventSeqs must be an array")
    output: list[int] = []
    has_range = False
    for entry in value:
        if not isinstance(entry, list):
            member = require_count(entry, "sourceEventSeqs member")
            if max_entries is not None and len(output) >= max_entries:
                raise SessionFormatError("sourceEventSeqs exceeds its event seq")
            output.append(member)
            continue
        if len(entry) != 2:
            raise SessionFormatError(
                "sourceEventSeqs range must be a [start, end] pair"
            )
        start = require_count(entry[0], "sourceEventSeqs range start")
        end = require_count(entry[1], "sourceEventSeqs range end")
        if end < start:
            raise SessionFormatError("sourceEventSeqs range exceeds its event seq")
        if max_entries is not None and (
            end >= max_entries or end - start + 1 > max_entries - len(output)
        ):
            raise SessionFormatError("sourceEventSeqs range exceeds its event seq")
        output.extend(range(start, end + 1))
        has_range = True
    if max_entries is not None:
        seen: set[int] = set()
        for member in output:
            if member >= max_entries or member in seen:
                raise SessionFormatError(
                    "sourceEventSeqs ranges must contain unique earlier seqs"
                )
            seen.add(member)
    if has_range and any(output[i] <= output[i - 1] for i in range(1, len(output))):
        raise SessionFormatError("sourceEventSeqs ranges must be strictly increasing")
    return output


# ---------------------------------------------------------------------------
# Logical header validation (exact keys per generation)
# ---------------------------------------------------------------------------

_LOGICAL_HEADER_REQUIRED = ("version", "id", "createdAt", "isSeeded", "delegationDepth")
_LOGICAL_HEADER_OPTIONAL = ("cwd", "parentSession", "origin", "agentPreset")


def _assert_logical_header_fields(header: Any, version: int, label: str) -> JsonObject:
    record = require_record(header, label)
    exact_keys(record, _LOGICAL_HEADER_REQUIRED, _LOGICAL_HEADER_OPTIONAL, label)
    if record["version"] != version:
        raise SessionFormatError(f"expected format v{version} header")
    if not isinstance(record["id"], str):
        raise SessionFormatError(f"format v{version} header id must be a string")
    require_count(record["createdAt"], f"format v{version} header createdAt")
    require_count(
        record["delegationDepth"], f"format v{version} header delegationDepth"
    )
    if not isinstance(record["isSeeded"], bool):
        raise SessionFormatError(f"format v{version} header isSeeded must be boolean")
    if isinstance(record.get("cwd"), str) and not os.path.isabs(record["cwd"]):
        raise SessionFormatError(f"format v{version} header cwd must be absolute")
    for key in ("cwd", "parentSession", "agentPreset"):
        if record.get(key) is not None and not isinstance(record[key], str):
            raise SessionFormatError(f"format v{version} header {key} must be a string")
    if record.get("origin") is not None and record["origin"] != "subagent":
        raise SessionFormatError(f'format v{version} header origin must be "subagent"')
    return record


def assert_v0_v1_header(header: Any, version: int) -> JsonObject:
    """Validate the logical header shared by released v0 and v1."""
    return _assert_logical_header_fields(header, version, f"format v{version} header")


def assert_v1_header(header: Any) -> JsonObject:
    """Validate one released-v1 logical header."""
    return assert_v0_v1_header(header, 1)


def assert_v2_header(header: Any) -> JsonObject:
    """Validate the exact logical header written by released v2."""
    return _assert_logical_header_fields(header, 2, "format v2 header")


def assert_v3_header(header: Any) -> JsonObject:
    """Validate v3 logical metadata with the released-v2 fields."""
    record = require_record(header, "format v3 header")
    if record.get("version") != 3:
        raise SessionFormatError("expected format v3 header")
    assert_v2_header({**record, "version": 2})
    return record


def assert_v4_header(header: Any) -> JsonObject:
    """Validate the exact native v4 logical header."""
    return _assert_logical_header_fields(header, 4, "format v4 header")


# ---------------------------------------------------------------------------
# Frozen event vocabularies
# ---------------------------------------------------------------------------

SURFACE_TYPES = frozenset(
    ["system/message", "user/message", "assistant/message", "tool/result"]
)
"""Event types that carry a surface operation in v3/v4 envelopes."""

CORE_CONTENT_TYPES = frozenset(
    ["text", "reasoning", "image", "file", "tool-call", "tool-result"]
)
"""Content block tags understood natively; anything else is namespaced."""


@dataclass(frozen=True)
class _PayloadDisposition:
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    opaque: tuple[str, ...] = ()


def _disp(
    required: list[str],
    optional: list[str] | None = None,
    opaque: list[str] | None = None,
) -> _PayloadDisposition:
    return _PayloadDisposition(
        tuple(required), tuple(optional or []), tuple(opaque or [])
    )


RELEASED_V0_EVENT_DISPOSITIONS: dict[str, _PayloadDisposition] = {
    "agent-preset/selected": _disp(["agentPreset"]),
    "agent/inbox/spliced": _disp(
        ["target", "start", "inserted"], ["removedCount", "outcome"]
    ),
    "approval/asked": _disp(["id", "toolName"], ["callId", "reason"]),
    "approval/decided": _disp(["id", "outcome"]),
    "approval/policy": _disp(["policy"], ["source"]),
    "assistant/chunk": _disp(["turn", "step", "chunk"]),
    "assistant/message": _disp(["turn", "step", "message"], ["usage", "interrupted"]),
    "command/done": _disp(["commandId", "kind"], ["text", "sourceEventSeq"]),
    "command/run": _disp(["commandId", "name", "source"], ["args"]),
    "compaction/end": _disp(["compactionId", "turn"], ["sourceCommandId", "error"]),
    "compaction/prune": _disp(["shadowedRange", "shadowedSeqs", "shadowedTokenCount"]),
    "compaction/start": _disp(["compactionId", "turn"], ["sourceCommandId"]),
    "compaction/summary": _disp(
        [
            "compactionId",
            "summary",
            "shadowedRange",
            "shadowedSeqs",
            "shadowedTokenCount",
            "provider",
            "model",
        ],
        ["sourceCommandId", "maxTokens", "usage", "rawOutput", "llmStreamCall"],
    ),
    "feedback/record": _disp(["text"]),
    "goal/change": _disp(
        ["kind", "version", "operation"],
        ["goal", "roundsStarted", "createdAt", "updatedAt", "cleared", "clearedAt"],
    ),
    "hook/invoked": _disp(["turn", "point", "dialect", "handlerId"], ["matcher"]),
    "hook/result": _disp(
        ["turn", "point", "handlerId", "decision", "durationMs"],
        ["exitCode", "stderrSummary"],
    ),
    "llm/retry": _disp(
        [
            "retryId",
            "turn",
            "step",
            "provider",
            "mode",
            "policyKey",
            "retry",
            "delayMs",
            "failure",
        ],
        ["maxRetries"],
    ),
    "llm/retry-started": _disp(["retryId", "turn", "step", "retry"]),
    "model/selection": _disp(["provider", "model"], ["reasoningEffort"]),
    "permission/preset": _disp(["preset"]),
    "plan/mode": _disp(["active"]),
    "request/context": _disp(["provider", "model"], ["contextWindow"]),
    "request/header": _disp(["header", "reason"], ["startsSeries"]),
    "sandbox/mode": _disp(["mode"], ["source"]),
    "schedule/change": _disp(
        ["version", "operation"], ["schedule", "id", "acceptedAt"]
    ),
    "session-log-deepseek/delivery-accepted": _disp(["sessionId", "throughSeq"]),
    "session/end-seed": _disp([]),
    "session/title": _disp(["title", "messageSeqs", "source"]),
    "session/title-llm-request": _disp(
        ["titleProvider", "messageSeqs", "route", "system", "messages", "maxTokens"]
    ),
    "step/end": _disp(["turn", "step"]),
    "step/start": _disp(["turn", "step"]),
    "subagent/descriptor": _disp(
        ["mode", "version", "provider"],
        [
            "label",
            "agentProvider",
            "agentModel",
            "agentReasoningEffort",
            "persona",
            "toolFilter",
        ],
    ),
    "subagent/model-selection-policy": _disp(["allowedModels"]),
    "team/member": _disp(["version", "teamId", "member"]),
    "team/message/delivered": _disp(["version", "teamId", "messageId", "targetId"]),
    "team/message/queued": _disp(["version", "teamId", "message"]),
    "team/task": _disp(["version", "teamId", "task"]),
    "todo/write": _disp(["todos"]),
    "tool-workflow/agent-end": _disp(["runId", "seq", "outcome"]),
    "tool-workflow/agent-start": _disp(["runId", "seq", "label", "childId"], ["phase"]),
    "tool-workflow/run-end": _disp(["runId", "stopReason"]),
    "tool-workflow/run-start": _disp(["runId", "name"]),
    "tool/call": _disp(["turn", "step", "callId", "name", "arguments"]),
    "tool/code-dispatch": _disp(
        [
            "rootCallId",
            "parentCallId",
            "subCallId",
            "name",
            "arguments",
            "isError",
            "content",
        ],
        [],
        ["arguments"],
    ),
    "tool/code-dispatch-start": _disp(
        ["rootCallId", "parentCallId", "subCallId", "name", "arguments"],
        [],
        ["arguments"],
    ),
    "tool/result": _disp(["turn", "step", "message"], ["error", "meta"], ["meta"]),
    "turn/end": _disp(["turn", "reason"]),
    "turn/start": _disp(["turn"]),
    "user/message": _disp(["role", "id", "content", "source"]),
    "web/deepseek-search-llm-request": _disp(["endpoint", "apiVersion", "body"]),
}
"""Exact top-level payload disposition frozen for every released-v0 event type."""

RELEASED_V0_EVENT_TYPES: tuple[str, ...] = tuple(sorted(RELEASED_V0_EVENT_DISPOSITIONS))

_RELEASED_V2_EXTRA: dict[str, _PayloadDisposition] = {
    "assistant/attempt": _disp(["turn", "step", "stream"]),
    "assistant/message": _disp(
        ["turn", "step", "message", "stream"], ["usage", "interrupted"]
    ),
    "session-log-deepseek/delivery-accepted": _disp(
        ["sessionId", "throughSeq"], ["sessionFormatVersion"]
    ),
    "session/end-seed": _disp([], ["inherited"]),
}

RELEASED_V2_EVENT_DISPOSITIONS: dict[str, _PayloadDisposition] = {
    **{
        key: value
        for key, value in RELEASED_V0_EVENT_DISPOSITIONS.items()
        if key
        not in (
            "assistant/chunk",
            "assistant/message",
            "session-log-deepseek/delivery-accepted",
            "session/end-seed",
        )
    },
    **_RELEASED_V2_EXTRA,
}
"""Exact top-level event and payload-member inventory frozen for released v2."""

RELEASED_V2_EVENT_TYPES: tuple[str, ...] = tuple(sorted(RELEASED_V2_EVENT_DISPOSITIONS))

RELEASED_V3_EVENT_TYPES: frozenset[str] = frozenset(
    [
        "agent-preset/selected",
        "agent/inbox/spliced",
        "approval/asked",
        "approval/decided",
        "approval/policy",
        "assistant/attempt",
        "assistant/message",
        "command/done",
        "command/run",
        "compaction/end",
        "compaction/prune",
        "compaction/start",
        "compaction/summary",
        "deliverables/presented",
        "feedback/message-delete",
        "feedback/message-put",
        "feedback/record",
        "goal/change",
        "hook/invoked",
        "hook/result",
        "image/offload",
        "llm/retry",
        "llm/retry-started",
        "model/selection",
        "permission/preset",
        "plan/mode",
        "request/context",
        "request/header",
        "sandbox/mode",
        "schedule/change",
        "session-log-deepseek/delivery-accepted",
        "session/end-seed",
        "session/title",
        "session/title-llm-request",
        "step/end",
        "step/start",
        "subagent/catalog",
        "subagent/descriptor",
        "subagent/model-selection-policy",
        "system/message",
        "team/member",
        "team/message/delivered",
        "team/message/queued",
        "team/task",
        "todo/write",
        "tool-workflow/agent-end",
        "tool-workflow/agent-start",
        "tool-workflow/run-end",
        "tool-workflow/run-start",
        "tool/call",
        "tool/ptc-dispatch",
        "tool/ptc-dispatch-start",
        "tool/result",
        "turn/end",
        "turn/start",
        "user/message",
        "web/deepseek-search-llm-request",
        "workspace/changes",
    ]
)
"""First-party event names understood by the released v3 reader (58)."""

KNOWN_SESSION_EVENT_TYPES: frozenset[str] = frozenset(
    set(RELEASED_V3_EVENT_TYPES)
    | {
        "developer/message",
        "job/registered",
        "job/settled",
        "working-directory/change",
    }
)
"""Every event type this build understands at format v4 (62)."""

# ---------------------------------------------------------------------------
# Artifacts, runs, envelopes
# ---------------------------------------------------------------------------


@dataclass
class SessionFormatArtifact:
    """One detached complete logical session artifact."""

    header: JsonObject
    inherited_event_count: int
    events: list[JsonObject] = field(default_factory=list)


@dataclass
class SessionFormatEventRun:
    """A codec-owned compact run migrations may consume without expanding."""

    run_type: str
    first_seq: int
    event_count: int
    expand_events: list[JsonObject] = field(default_factory=list)

    def expand(self) -> list[JsonObject]:
        """Expand the run into its events, in order."""
        return list(self.expand_events)


@dataclass
class PackedAssistantChunkRun(SessionFormatEventRun):
    """A released packed assistant row retained until v1->v2 embeds it."""

    turn: int = 0
    step: int = 0
    last_seq: int = 0
    last_time: int = 0
    stream: JsonObject = field(default_factory=dict)

    def expand(self) -> list[JsonObject]:
        """Expand packed chunks into ``assistant/chunk`` events."""
        gaps = self.stream["dt"]
        members = (
            self.stream["args"]
            if self.stream["type"] == "tool-call-chunks"
            else self.stream["texts"]
        )
        time = self.stream["time0"]
        events: list[JsonObject] = []
        for index, member in enumerate(members):
            if index > 0:
                time += gaps[index - 1]
            if self.stream["type"] == "text-chunks":
                chunk = {
                    "type": "text-delta",
                    "index": self.stream["index"],
                    "text": member,
                }
            elif self.stream["type"] == "reasoning-chunks":
                chunk = {
                    "type": "reasoning-delta",
                    "index": self.stream["index"],
                    "text": member,
                }
            else:
                chunk = {
                    "type": "tool-call-delta",
                    "index": self.stream["index"],
                    "id": self.stream["id"],
                    **({"name": self.stream["name"]} if "name" in self.stream else {}),
                    "argumentsDelta": member,
                }
            events.append(
                {
                    "type": "assistant/chunk",
                    "seq": self.first_seq + index,
                    "time": time,
                    "data": {"turn": self.turn, "step": self.step, "chunk": chunk},
                }
            )
        return events


_EVENT_REQUIRED = ("type", "seq", "time", "data")
_EVENT_OPTIONAL = ("ignorable", "sourceEventSeqs", "surfaceOp")
_EVENT_KEYS = frozenset((*_EVENT_REQUIRED, *_EVENT_OPTIONAL))


def decode_envelope_row(row: Any, row_index: int, label: str) -> JsonObject:
    """Decode one physical event row's envelope (exact keys, safe fields).

    Shared by the v2 framing and by the load path's envelope-only read of
    current-version files. Does not enforce seq density or payload shapes.
    """
    record = require_record(row, f"{label} row {row_index}")
    missing = next((key for key in _EVENT_REQUIRED if key not in record), None)
    if missing is not None:
        raise SessionFormatError(
            f"{label} row {row_index} lacks required field {missing}"
        )
    unexpected = next((key for key in record if key not in _EVENT_KEYS), None)
    if unexpected is not None:
        raise SessionFormatError(
            f"{label} row {row_index} has unexpected field {unexpected}"
        )
    if not isinstance(record["type"], str):
        raise SessionFormatError(f"{label} row {row_index} type must be a string")
    require_safe_integer(record["time"], f"{label} row {row_index} time")
    if record.get("ignorable") is not None and record["ignorable"] is not True:
        raise SessionFormatError(
            f"{label} row {row_index} ignorable must be true when present"
        )
    if record.get("sourceEventSeqs") is None:
        return record
    seq = require_count(record["seq"], f"{label} row {row_index} seq")
    return {
        **record,
        "sourceEventSeqs": decode_seq_ranges(record["sourceEventSeqs"], seq),
    }


def encode_envelope_event(event: JsonObject) -> JsonObject:
    """Encode one logical event as one physical record (range-encoded)."""
    if event.get("sourceEventSeqs") is None:
        return dict(event)
    return {
        **event,
        "sourceEventSeqs": encode_seq_ranges(
            [int(seq) for seq in event["sourceEventSeqs"]]
        ),
    }


def assert_released_event_payload(event: JsonObject, version: int) -> None:
    """Validate one exact known payload after legacy normalization.

    Applies the frozen disposition's exact member rules plus the targeted
    semantic checks the migration edges depend on (retry and compaction
    identity fields, subagent descriptor versions). The full per-payload
    semantic validation of the reference implementation is not ported here;
    it belongs to the event-vocabulary work that owns the 62-type registry.
    """
    event_type = event.get("type")
    disposition = RELEASED_V0_EVENT_DISPOSITIONS.get(str(event_type))
    if disposition is None:
        raise SessionFormatUnsupportedMigrationError(
            f"format v{version} contains unknown historical event type "
            f"{json.dumps(event_type)} at seq {event.get('seq')}; migration refuses "
            "unknown historical events even when ignorable"
        )
    data = require_record(event.get("data"), f"{event_type} {event.get('seq')} data")
    if event_type == "subagent/descriptor" and data.get("version") != 3:
        descriptor_version = require_count(
            data.get("version"), f"{event_type} {event.get('seq')} version"
        )
        if version == 0:
            raise SessionFormatUnsupportedMigrationError(
                f"{event_type} {event.get('seq')} uses unsupported descriptor "
                f"version {descriptor_version}"
            )
        return
    optional = disposition.optional
    if version == 1 and event_type == "session-log-deepseek/delivery-accepted":
        optional = (*optional, "sessionFormatVersion")
    exact_keys(
        data,
        disposition.required,
        optional,
        f"{event_type} {event.get('seq')} data",
    )
    if event_type == "llm/retry":
        retry_id = data.get("retryId")
        if not isinstance(retry_id, str) or not retry_id:
            raise SessionFormatError(
                f"llm/retry {event.get('seq')} retryId must be a non-empty string"
            )
    if event_type in ("compaction/start", "compaction/summary", "compaction/end"):
        compaction_id = data.get("compactionId")
        if not isinstance(compaction_id, str) or not compaction_id:
            raise SessionFormatError(
                f"{event_type} {event.get('seq')} compactionId must be "
                "a non-empty string"
            )


def assert_released_surface_metadata(
    record: JsonObject, seq: int, event_type: str, assistant_sources: str
) -> None:
    """Validate shared-layout surface references for one released generation."""
    sources = record.get("sourceEventSeqs")
    if (
        event_type == "assistant/message"
        and sources is not None
        and assistant_sources == "forbid-assistant"
    ):
        raise SessionFormatError(
            f"assistant/message {seq} retains obsolete chunk references"
        )
    if sources is not None:
        if not isinstance(sources, list):
            raise SessionFormatError(
                f"{event_type} {seq} sourceEventSeqs must be an array"
            )
        seen: set[int] = set()
        for source in sources:
            current = require_count(
                source, f"{event_type} {seq} sourceEventSeqs member"
            )
            if current >= seq or current in seen:
                raise SessionFormatError(
                    f"{event_type} {seq} sourceEventSeqs must be unique earlier seqs"
                )
            seen.add(current)
        if len(sources) == 0 and event_type != "assistant/message":
            raise SessionFormatError(
                f"{event_type} {seq} sourceEventSeqs must be non-empty"
            )
    operation = record.get("surfaceOp")
    if operation is None or operation == "append":
        return
    replacement = require_record(operation, f"{event_type} {seq} surfaceOp")
    exact_keys(replacement, ["op", "start", "end"], [], f"{event_type} {seq} surfaceOp")
    if replacement["op"] != "replace":
        raise SessionFormatError(f"{event_type} {seq} surfaceOp must replace")
    start = require_count(replacement["start"], f"{event_type} {seq} surface start")
    end = require_count(replacement["end"], f"{event_type} {seq} surface end")
    if start >= seq or end >= seq:
        raise SessionFormatError(
            f"{event_type} {seq} has an invalid surface replacement"
        )


# ---------------------------------------------------------------------------
# Delivery-acceptance watermark fold
# ---------------------------------------------------------------------------


def validate_delivery_accepted(event: JsonObject, current_version: int) -> str | None:
    """Validate delivery generation and active-generation coordinates.

    Returns the active delivery's non-empty session id, or ``None`` for other
    events and generations. The upload behavior itself is out of scope; this
    fold exists so migrations can validate delivery markers.
    """
    if event.get("type") != "session-log-deepseek/delivery-accepted":
        return None
    data = require_record(event.get("data"), "delivery-accepted data")
    version = require_count(
        data.get("sessionFormatVersion", 0), "delivery sessionFormatVersion"
    )
    if version != current_version:
        return None
    through_seq = require_count(data.get("throughSeq"), "delivery throughSeq")
    if through_seq >= event["seq"]:
        raise SessionFormatError("delivery throughSeq must precede its marker")
    session_id = data.get("sessionId")
    if not isinstance(session_id, str) or not session_id:
        raise SessionFormatError("delivery requires a nonempty Session id")
    return session_id


def accepted_through(
    events: list[JsonObject], header_version: int, session_id: str
) -> int:
    """Highest confirmed sequence for one exact session format generation.

    Folds ``session-log-deepseek/delivery-accepted`` markers whose format
    version matches the session and whose session id matches. Returns ``-1``
    before any accepted request. A malformed watermark raises
    :class:`SessionFormatError`.
    """
    through_seq = -1
    for event in events:
        if event.get("type") != "session-log-deepseek/delivery-accepted":
            continue
        data = require_record(event.get("data"), "delivery-accepted data")
        accepted_version = data.get("sessionFormatVersion", 0)
        if not _is_safe_integer(accepted_version) or accepted_version < 0:
            raise SessionFormatError(
                f"malformed acceptance format version at seq {event.get('seq')}"
            )
        if accepted_version != header_version:
            continue
        accepted_seq = data.get("throughSeq")
        if not _is_safe_integer(accepted_seq):
            raise SessionFormatError(
                f"malformed acceptance watermark at seq {event.get('seq')}"
            )
        if (
            not isinstance(data.get("sessionId"), str)
            or not data["sessionId"]
            or accepted_seq >= event["seq"]
        ):
            raise SessionFormatError(
                f"malformed acceptance watermark at seq {event.get('seq')}"
            )
        if data["sessionId"] != session_id:
            continue
        through_seq = max(through_seq, cast(int, accepted_seq))
    return through_seq


# ---------------------------------------------------------------------------
# Physical codecs
# ---------------------------------------------------------------------------

PACKED_TAGS = frozenset(["text-chunks", "reasoning-chunks", "tool-call-chunks"])

_V01_HEADER_REQUIRED = ("type", "version", "id", "createdAt", "delegationDepth")
_V01_HEADER_OPTIONAL = (
    "cwd",
    "parentSession",
    "seedLength",
    "origin",
    "agentPreset",
)

_V2_HEADER_REQUIRED = (
    "type",
    "version",
    "id",
    "createdAt",
    "isSeeded",
    "delegationDepth",
)
_V2_HEADER_OPTIONAL = ("cwd", "parentSession", "origin", "agentPreset")


class _Emit:
    """Synchronous output channel for decoders and migration stages."""

    def emit_event(self, event: JsonObject) -> None:
        """Deliver one settled event to the next stage before returning."""

    def emit_run(self, run: SessionFormatEventRun) -> None:
        """Deliver one compact event run to the next stage before returning."""


class _Collector(_Emit):
    """Migration output context that expands compact runs into events."""

    def __init__(self) -> None:
        self.values: list[JsonObject] = []

    def emit_event(self, event: JsonObject) -> None:
        self.values.append(event)

    def emit_run(self, run: SessionFormatEventRun) -> None:
        self.values.extend(run.expand())


class _Decoder:
    """Stateful physical-row decoder used by streaming restores."""

    header: JsonObject
    header_inherited_event_count: int | None

    def decode_row(self, row: Any, emit: _Emit) -> None:
        """Decode one physical row and synchronously emit its events or run."""

    def finish(self) -> int:
        """Finish row validation and return the exact inherited cut."""
        raise NotImplementedError("decoder must implement finish")


def _decode_packed_run(
    row: JsonObject, row_type: str, row_index: int
) -> PackedAssistantChunkRun:
    label = f"released {row_type} row {row_index}"
    exact_keys(row, ["type", "seq0", "time0", "data"], [], label)
    seq0 = require_count(row["seq0"], f"{label} seq0")
    time0 = require_safe_integer(row["time0"], f"{label} time0")
    data = require_record(row["data"], f"{label} data")
    is_tool = row_type == "tool-call-chunks"
    exact_keys(
        data,
        (
            ["turn", "step", "index", "id", "dt", "args"]
            if is_tool
            else ["turn", "step", "index", "dt", "texts"]
        ),
        ["name"] if is_tool else [],
        f"{label} data",
    )
    payload = data["args"] if is_tool else data["texts"]
    if (
        not isinstance(payload, list)
        or not payload
        or any(not isinstance(member, str) for member in payload)
    ):
        raise SessionFormatError(f"{label} payload must be a non-empty string array")
    gaps = data["dt"]
    if not isinstance(gaps, list) or len(gaps) != len(payload) - 1:
        raise SessionFormatError(f"{label} dt length must match its payload")
    last_time = time0
    for gap in gaps:
        valid_gap = require_safe_integer(gap, f"{label} dt member")
        last_time = require_safe_integer(last_time + valid_gap, f"{label} member time")
    turn = require_count(data["turn"], f"{label} turn")
    step = require_count(data["step"], f"{label} step")
    chunk_index = require_count(data["index"], f"{label} index")
    if is_tool and (
        not isinstance(data["id"], str)
        or not data["id"]
        or ("name" in data and not isinstance(data["name"], str))
    ):
        raise SessionFormatError(f"{label} id and optional name must be strings")
    last_seq = require_count(seq0 + len(payload) - 1, f"{label} final seq")
    stream: JsonObject = (
        {
            "type": row_type,
            "time0": time0,
            "index": chunk_index,
            "dt": gaps,
            "id": data["id"],
            **({"name": data["name"]} if "name" in data else {}),
            "args": payload,
        }
        if is_tool
        else {
            "type": row_type,
            "time0": time0,
            "index": chunk_index,
            "dt": gaps,
            "texts": payload,
        }
    )
    return PackedAssistantChunkRun(
        run_type="released-assistant-chunks",
        first_seq=seq0,
        event_count=len(payload),
        turn=turn,
        step=step,
        last_seq=last_seq,
        last_time=last_time,
        stream=stream,
    )


class _V01Decoder(_Decoder):
    """Row scanner for the shared released v0/v1 layout."""

    def __init__(self, header_value: Any, version: int, recovery: str) -> None:
        physical = self._decode_physical_header(header_value, version)
        self.header = physical[0]
        self.header_inherited_event_count = physical[1]
        self._recoverable = recovery == "recoverable"
        self._row_index = 0
        self._event_count = 0
        self._issue: SessionFormatError | None = None
        self._version = version

    @staticmethod
    def _decode_physical_header(value: Any, version: int) -> tuple[JsonObject, int]:
        record = require_record(value, f"released v{version} physical header")
        exact_keys(
            record,
            _V01_HEADER_REQUIRED,
            _V01_HEADER_OPTIONAL,
            f"released v{version} physical header",
        )
        if record["type"] != "session" or record["version"] != version:
            raise SessionFormatError(
                f"expected released v{version} physical Session header"
            )
        if not isinstance(record["id"], str):
            raise SessionFormatError(f"released v{version} header id must be a string")
        created_at = require_count(
            record["createdAt"], f"released v{version} header createdAt"
        )
        delegation_depth = require_count(
            record["delegationDepth"],
            f"released v{version} header delegationDepth",
        )
        seed_length = (
            0
            if record.get("seedLength") is None
            else require_count(
                record["seedLength"], f"released v{version} header seedLength"
            )
        )
        for key in ("cwd", "parentSession", "agentPreset"):
            if record.get(key) is not None and not isinstance(record[key], str):
                raise SessionFormatError(
                    f"released v{version} header {key} must be a string"
                )
        if record.get("origin") is not None and record["origin"] != "subagent":
            raise SessionFormatError(
                f'released v{version} header origin must be "subagent"'
            )
        header: JsonObject = {
            "version": version,
            "id": record["id"],
            "createdAt": created_at,
            **({"cwd": record["cwd"]} if "cwd" in record else {}),
            **(
                {"parentSession": record["parentSession"]}
                if "parentSession" in record
                else {}
            ),
            "isSeeded": "seedLength" in record,
            **({"origin": record["origin"]} if "origin" in record else {}),
            "delegationDepth": delegation_depth,
            **(
                {"agentPreset": record["agentPreset"]}
                if "agentPreset" in record
                else {}
            ),
        }
        assert_v0_v1_header(header, version)
        return header, seed_length

    def decode_row(self, row: Any, emit: _Emit) -> None:
        current_row = self._row_index
        self._row_index += 1
        packed = False
        decoded: JsonObject | PackedAssistantChunkRun
        try:
            record = require_record(row, f"released Session row {current_row}")
            row_type = record.get("type")
            if isinstance(row_type, str) and row_type in PACKED_TAGS:
                packed = True
                decoded = _decode_packed_run(record, row_type, current_row)
            else:
                decoded = decode_envelope_row(record, current_row, "released Session")
        except SessionFormatError as error:
            if not self._recoverable:
                raise
            if self._issue is None:
                self._issue = error
            return
        # Reachable only through programmatic misuse: decode_row takes Any
        # and is a catalog entry point, so a non-JSON row (for example a
        # packed row carrying a non-serializable key, which fails inside
        # exact_keys) lands here. JSON rows cannot reach it: their keys are
        # always strings, and every other failure inside the try raises
        # SessionFormatError.
        except (TypeError, ValueError) as error:
            current = SessionFormatError(
                f"released Session row {current_row} is malformed"
            )
            current.__cause__ = error
            if not self._recoverable:
                raise current
            if self._issue is None:
                self._issue = current
            return
        if self._issue is not None:
            if not packed and decoded["type"] == "turn/end":  # type: ignore[index]
                raise self._issue
            return
        seq = (
            decoded.first_seq  # type: ignore[union-attr]
            if packed
            else decoded["seq"]  # type: ignore[index]
        )
        if seq != self._event_count:
            gap = SessionFormatError(
                f"released Session row {current_row} has seq gap "
                f"(expected {self._event_count}, got {seq})"
            )
            if not self._recoverable:
                raise gap
            self._issue = gap
            if not packed and decoded["type"] == "turn/end":  # type: ignore[index]
                raise gap
            return
        if packed:
            run = decoded  # type: ignore[assignment]
            self._event_count += run.event_count  # type: ignore[union-attr]
            emit.emit_run(run)  # type: ignore[arg-type]
        else:
            self._event_count += 1
            emit.emit_event(decoded)  # type: ignore[arg-type]

    def finish(self) -> int:
        inherited = self.header_inherited_event_count
        if inherited is not None and inherited > self._event_count:
            raise SessionFormatError(
                "Session inheritedEventCount exceeds its event count"
            )
        return inherited if inherited is not None else 0


class _V01Codec:
    """Frozen physical JSON codec for the released v0/v1 shared layout."""

    def __init__(self, version: int) -> None:
        if version not in (0, 1):
            raise SessionFormatError("released v0/v1 codec requires version 0 or 1")
        self.version = version

    def decode_header(self, value: Any) -> JsonObject:
        return _V01Decoder._decode_physical_header(value, self.version)[0]

    def create_decoder(self, header_value: Any, recovery: str) -> _Decoder:
        return _V01Decoder(header_value, self.version, recovery)


class _V2Decoder(_Decoder):
    """Row scanner for the released v2 framing (also reused by v3/v4)."""

    def __init__(self, header_value: Any, recovery: str) -> None:
        self.header = self._decode_physical_header(header_value)
        self.header_inherited_event_count = None
        self._recoverable = recovery == "recoverable"
        self._row_index = 0
        self._event_count = 0
        self._issue: SessionFormatError | None = None
        self._inherited: int | None = None

    @staticmethod
    def _decode_physical_header(value: Any) -> JsonObject:
        record = require_record(value, "released v2 physical header")
        exact_keys(
            record,
            _V2_HEADER_REQUIRED,
            _V2_HEADER_OPTIONAL,
            "released v2 physical header",
        )
        if record["type"] != "session" or record["version"] != 2:
            raise SessionFormatError("expected released v2 physical Session header")
        if not isinstance(record["id"], str):
            raise SessionFormatError("released v2 header id must be a string")
        created_at = require_count(record["createdAt"], "released v2 header createdAt")
        delegation_depth = require_count(
            record["delegationDepth"], "released v2 header delegationDepth"
        )
        if not isinstance(record["isSeeded"], bool):
            raise SessionFormatError("released v2 header isSeeded must be boolean")
        for key in ("cwd", "parentSession", "agentPreset"):
            if record.get(key) is not None and not isinstance(record[key], str):
                raise SessionFormatError(f"released v2 header {key} must be a string")
        if record.get("origin") is not None and record["origin"] != "subagent":
            raise SessionFormatError('released v2 header origin must be "subagent"')
        header: JsonObject = {
            "version": 2,
            "id": record["id"],
            "createdAt": created_at,
            **({"cwd": record["cwd"]} if "cwd" in record else {}),
            **(
                {"parentSession": record["parentSession"]}
                if "parentSession" in record
                else {}
            ),
            "isSeeded": record["isSeeded"],
            **({"origin": record["origin"]} if "origin" in record else {}),
            "delegationDepth": delegation_depth,
            **(
                {"agentPreset": record["agentPreset"]}
                if "agentPreset" in record
                else {}
            ),
        }
        assert_v2_header(header)
        return header

    def decode_row(self, row: Any, emit: _Emit) -> None:
        current_row = self._row_index
        self._row_index += 1
        try:
            event = decode_envelope_row(row, current_row, "released v2")
        except SessionFormatError as error:
            if self._recoverable:
                if self._issue is None:
                    self._issue = error
                return
            raise
        if self._issue is not None:
            if event["type"] == "turn/end":
                raise self._issue
            return
        if event["seq"] != self._event_count:
            gap = SessionFormatError(
                f"released v2 row {current_row} has seq gap "
                f"(expected {self._event_count}, got {event['seq']})"
            )
            if self._recoverable:
                self._issue = gap
                if event["type"] == "turn/end":
                    raise gap
                return
            raise gap
        self._event_count += 1
        if event["type"] == "session/end-seed":
            data = require_record(
                event["data"], f"session/end-seed {event['seq']} data"
            )
            if data.get("inherited") is True:
                self._inherited = event["seq"]
        emit.emit_event(event)

    def finish(self) -> int:
        if self.header["isSeeded"] and self._inherited is None:
            raise SessionFormatError(
                "released v2 seeded Session lacks an inherited end-seed marker"
            )
        if not self.header["isSeeded"] and self._inherited is not None:
            raise SessionFormatError(
                "released v2 unseeded Session contains an inherited end-seed marker"
            )
        return self._inherited if self._inherited is not None else 0


def _encode_v2_header(header: JsonObject, inherited_event_count: int) -> JsonObject:
    assert_v2_header(header)
    cut = require_count(inherited_event_count, "format v2 inherited event count")
    if not header["isSeeded"] and cut != 0:
        raise SessionFormatError("unseeded format v2 Session has inherited events")
    return {
        "type": "session",
        "version": 2,
        "id": header["id"],
        "createdAt": header["createdAt"],
        **({"cwd": header["cwd"]} if "cwd" in header else {}),
        **(
            {"parentSession": header["parentSession"]}
            if "parentSession" in header
            else {}
        ),
        "isSeeded": header["isSeeded"],
        **({"origin": header["origin"]} if "origin" in header else {}),
        "delegationDepth": header["delegationDepth"],
        **({"agentPreset": header["agentPreset"]} if "agentPreset" in header else {}),
    }


class _V2Codec:
    """Frozen physical JSON codec for released v2 (also the current encoder)."""

    version = 2

    def decode_header(self, value: Any) -> JsonObject:
        return _V2Decoder._decode_physical_header(value)

    def create_decoder(self, header_value: Any, recovery: str) -> _Decoder:
        return _V2Decoder(header_value, recovery)

    def encode_header(
        self, header: JsonObject, inherited_event_count: int
    ) -> JsonObject:
        return _encode_v2_header(header, inherited_event_count)

    def encode_event(self, event: JsonObject) -> JsonObject:
        return encode_envelope_event(event)


# ---------------------------------------------------------------------------
# Native v4 admission (developer / retired syntax / system / tool-role)
# ---------------------------------------------------------------------------


def _map_event_messages(
    event: JsonObject,
    transform: Callable[[JsonObject], JsonObject],
) -> JsonObject:
    """Visit only messages carried by first-party event payloads."""
    data = event.get("data")
    if not is_json_object(data):
        return event
    if event.get("type") == "user/message":
        message = transform(data)
        return event if message is data else {**event, "data": message}
    if event.get("type") in (
        "developer/message",
        "system/message",
        "assistant/message",
        "tool/result",
    ):
        if not is_json_object(data.get("message")):
            raise SessionFormatError(event["type"] + " requires a message")
        message = transform(data["message"])
        if message is data["message"]:
            return event
        return {**event, "data": {**data, "message": message}}
    key = (
        "inserted"
        if event.get("type") == "agent/inbox/spliced"
        else ("messages" if event.get("type") == "session/title-llm-request" else None)
    )
    if key is None:
        return event
    messages = data.get(key)
    if not isinstance(messages, list):
        raise SessionFormatError(event["type"] + " requires message array")
    mapped = []
    for message in messages:
        if not is_json_object(message):
            raise SessionFormatError(event["type"] + " requires message objects")
        mapped.append(transform(message))
    if all(mapped[i] is messages[i] for i in range(len(mapped))):
        return event
    return {**event, "data": {**data, key: mapped}}


def _assert_tool_change(block: Any, developer: bool) -> None:
    if not is_json_object(block) or block.get("type") not in (
        "tool-addition",
        "tool-removal",
    ):
        return
    if not developer:
        raise SessionFormatError(f"format v4 {block['type']} requires developer role")
    if not isinstance(block.get("toolName"), str) or not block["toolName"]:
        raise SessionFormatError(
            f"format v4 {block['type']} requires a nonempty toolName"
        )
    if block["type"] == "tool-addition" and "tool" in block:
        raise SessionFormatError(
            "format v4 tool-addition must omit inline tool definitions"
        )


def _assert_content_blocks(value: Any, developer: bool = False) -> None:
    if isinstance(value, list):
        for block in value:
            _assert_tool_change(block, developer)


def assert_v4_developer_data(event: JsonObject) -> None:
    """Validate native v4 developer messages and deferred tool schemas."""
    data = event.get("data")
    if not is_json_object(data):
        if event.get("type") == "developer/message":
            raise SessionFormatError(
                "format v4 developer/message data must be an object"
            )
        return
    if event.get("type") == "developer/message":
        message = data.get("message")
        if not is_json_object(message) or message.get("role") != "developer":
            raise SessionFormatError(
                "format v4 developer/message requires turn, step, and a "
                "developer message"
            )
        for field_name in ("turn", "step"):
            if (
                require_count(data.get(field_name), f"developer/message {field_name}")
                == 0
            ):
                raise SessionFormatError(
                    f"developer/message {field_name} must be positive"
                )
        _assert_content_blocks(message.get("content"), True)
        source = message.get("source")
        if (
            not isinstance(message.get("id"), str)
            or not message["id"]
            or not isinstance(message.get("content"), list)
            or not is_json_object(source)
            or not isinstance(source.get("kind"), str)
            or not source["kind"]
            or source["kind"] == "plugin"
        ):
            raise SessionFormatError(
                "format v4 developer message requires id, role, content, and "
                "a producer-owned source"
            )
        content = message["content"]
        has_additions = any(
            is_json_object(block) and block.get("type") == "tool-addition"
            for block in content
        )
        if has_additions:
            require_count(data.get("headerSeq"), "developer/message headerSeq")
        elif "headerSeq" in data:
            raise SessionFormatError(
                "format v4 developer/message must omit headerSeq without tool additions"
            )
        return
    if event.get("type") == "user/message":
        message = data
        # pragma: no cover - defensive only: data is proven a JSON object
        # above, and message is data itself, so this False edge cannot fire.
        if is_json_object(message):  # pragma: no cover
            if message.get("role") == "developer":
                raise SessionFormatError(
                    "format v4 developer/message and developer role must occur together"
                )
            _assert_content_blocks(message.get("content"))
    elif event.get("type") in (
        "system/message",
        "assistant/message",
        "tool/result",
    ):
        inner = data.get("message")
        if is_json_object(inner):
            if inner.get("role") == "developer":
                raise SessionFormatError(
                    "format v4 developer/message and developer role must occur together"
                )
            _assert_content_blocks(inner.get("content"))
    elif event.get("type") in ("agent/inbox/spliced", "session/title-llm-request"):
        messages = data.get(
            "inserted" if event["type"] == "agent/inbox/spliced" else "messages"
        )
        if isinstance(messages, list):
            for message in messages:
                if is_json_object(message):
                    if message.get("role") == "developer":
                        raise SessionFormatError(
                            "format v4 developer/message and developer role "
                            "must occur together"
                        )
                    _assert_content_blocks(message.get("content"))
    if event.get("type") == "compaction/summary":
        _assert_content_blocks(data.get("summary"))
        _assert_content_blocks(data.get("rawOutput"))
    if event.get("type") in ("assistant/message", "assistant/attempt") and isinstance(
        data.get("stream"), list
    ):
        for entry in data["stream"]:
            if not is_json_object(entry) or entry.get("type") != "chunk":
                continue
            chunk = entry.get("chunk")
            if not is_json_object(chunk):
                continue
            if chunk.get("type") == "block-end":
                _assert_tool_change(chunk.get("block"), False)
            if chunk.get("type") == "block-start" and chunk.get("blockType") in (
                "tool-addition",
                "tool-removal",
            ):
                raise SessionFormatError(
                    "format v4 tool-change blocks require developer role"
                )
    if event.get("type") == "request/header" and is_json_object(data.get("header")):
        tools = data["header"].get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if not is_json_object(tool) or "deferLoading" not in tool:
                    continue
                if tool["deferLoading"] is not True:
                    raise SessionFormatError(
                        "format v4 tool deferLoading must be true when present"
                    )


def assert_v4_retired_syntax(row: Any) -> None:
    """Refuse retired headers, PTC tags, and tool-result blocks in content slots."""
    if not is_json_object(row):
        return
    if row.get("type") in ("tool/code-dispatch-start", "tool/code-dispatch") and (
        row.get("ignorable") is not True
    ):
        raise SessionFormatUnsupportedMigrationError(
            f"format v4 rejects retired event type {row['type']}"
        )
    data = row.get("data")
    if row.get("type") == "request/header":
        if not is_json_object(data) or not is_json_object(data.get("header")):
            raise SessionFormatError(
                "format v4 request/header requires data and header objects"
            )
        if "system" in data["header"]:
            raise SessionFormatError(
                "format v4 request/header rejects retired header.system"
            )
        return
    if not is_json_object(data):
        return
    subject = f"format v4 {row.get('type')} at seq {row.get('seq')} content"

    def assert_block(block: Any) -> None:
        if is_json_object(block) and block.get("type") == "tool-result":
            raise SessionFormatError(
                f"{subject} must not contain a released tool-result wrapper"
            )

    def assert_content(content: Any) -> None:
        if isinstance(content, list):
            for block in content:
                assert_block(block)

    def assert_message_content(message: Any) -> None:
        if is_json_object(message):
            assert_content(message.get("content"))

    row_type = row.get("type")
    if row_type == "user/message":
        assert_content(data.get("content"))
    elif row_type in (
        "developer/message",
        "assistant/message",
        "team/message/queued",
    ):
        assert_message_content(data.get("message"))
    elif row_type in ("agent/inbox/spliced", "session/title-llm-request"):
        messages = data.get(
            "inserted" if row_type == "agent/inbox/spliced" else "messages"
        )
        if isinstance(messages, list):
            for message in messages:
                assert_message_content(message)
    elif row_type == "compaction/summary":
        assert_content(data.get("summary"))
        assert_content(data.get("rawOutput"))
    elif row_type == "tool/ptc-dispatch":
        assert_content(data.get("content"))
    if row_type in ("assistant/message", "assistant/attempt") and isinstance(
        data.get("stream"), list
    ):
        for entry in data["stream"]:
            if not is_json_object(entry) or entry.get("type") != "chunk":
                continue
            chunk = entry.get("chunk")
            if not is_json_object(chunk):
                continue
            if chunk.get("type") == "block-end":
                assert_block(chunk.get("block"))
            if (
                chunk.get("type") == "block-start"
                and chunk.get("blockType") == "tool-result"
            ):
                raise SessionFormatError(
                    f"{subject} must not contain a released tool-result wrapper"
                )


def assert_v4_system_message_fields(row: Any) -> None:
    """Validate system fields independently of historical message forms."""
    if not is_json_object(row) or row.get("type") != "system/message":
        return
    data = require_record(row.get("data"), "system/message data")
    for field_name in ("turn", "step"):
        if require_count(data.get(field_name), f"system/message {field_name}") == 0:
            raise SessionFormatError(
                f"format v4 system/message {field_name} must be positive"
            )
    message = require_record(data.get("message"), "system message")
    if not isinstance(message.get("id"), str) or not message["id"]:
        raise SessionFormatError("format v4 system message requires a string id")
    if message.get("role") != "system":
        raise SessionFormatError("format v4 system message requires system role")
    content = message.get("content")
    if not isinstance(content, list):
        raise SessionFormatError("format v4 system message requires content array")
    for value in content:
        block = require_record(value, "system content block")
        if not isinstance(block.get("type"), str) or not block["type"]:
            raise SessionFormatError("format v4 system content requires a string type")
        block_type = block["type"]
        if block_type in ("text", "reasoning"):
            if not isinstance(block.get("text"), str):
                raise SessionFormatError(
                    "format v4 system content text requires a string"
                )
        elif block_type == "tool-call":
            for key in ("id", "name"):
                if not isinstance(block.get(key), str) or not block[key]:
                    raise SessionFormatError(
                        f"format v4 system tool-call {key} requires a string"
                    )
            if not isinstance(block.get("arguments"), str):
                raise SessionFormatError(
                    "format v4 system tool-call arguments requires a string"
                )
        elif block_type == "image":
            attachment = require_record(
                block.get("attachment"), "system image attachment"
            )
            if (
                not isinstance(attachment.get("attachmentId"), str)
                or not attachment["attachmentId"]
            ):
                raise SessionFormatError(
                    "format v4 system image attachmentId requires a string"
                )
            if attachment.get("mediaType") not in (
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
            ):
                raise SessionFormatError(
                    "format v4 system image requires a supported mediaType"
                )
            require_count(attachment.get("bytes"), "system image bytes")
            for key in ("width", "height"):
                if require_count(attachment.get(key), f"system image {key}") == 0:
                    raise SessionFormatError(
                        f"format v4 system image {key} must be positive"
                    )
            if attachment.get("name") is not None and not isinstance(
                attachment["name"], str
            ):
                raise SessionFormatError(
                    "format v4 system image name requires a string"
                )
            if attachment.get("originalDimensions") is not None:
                original = require_record(
                    attachment["originalDimensions"],
                    "system image originalDimensions",
                )
                for key in ("width", "height"):
                    if (
                        require_count(original.get(key), f"system image original {key}")
                        == 0
                    ):
                        raise SessionFormatError(
                            f"format v4 system image original {key} must be positive"
                        )
        elif block_type == "tool-result":
            raise SessionFormatError(
                "format v4 system content rejects retired tool-result wrappers"
            )


def assert_v4_tool_result_message(event: JsonObject) -> None:
    """Validate the native v4 first-class tool-role message of one row."""
    if event.get("type") != "tool/result":
        return
    subject = f"format v4 tool/result at seq {event.get('seq')}"
    data = require_record(event.get("data"), f"{subject} data")
    message = require_record(data.get("message"), f"{subject} message")
    message_id = message.get("id")
    role = message.get("role")
    tool_call_id = message.get("toolCallId")
    source = message.get("source")
    is_error = message.get("isError")
    content = message.get("content")
    source_call_id = source.get("callId") if is_json_object(source) else None
    if not isinstance(message_id, str) or not message_id:
        raise SessionFormatError(
            f"{subject} requires a first-class message with a string id"
        )
    if role != "tool":
        raise SessionFormatError(f"{subject} requires a tool-role message")
    if (
        not isinstance(tool_call_id, str)
        or not tool_call_id
        or source_call_id != tool_call_id
    ):
        raise SessionFormatError(
            f"{subject} requires toolCallId matching its tool source"
        )
    if not is_json_object(source) or source.get("kind") != "tool":
        raise SessionFormatError(f"{subject} requires a tool source")
    if not isinstance(content, list):
        raise SessionFormatError(f"{subject} requires array content")
    if any(
        is_json_object(block) and block.get("type") == "tool-result"
        for block in content
    ):
        raise SessionFormatError(
            f"{subject} content must not contain a released tool-result wrapper"
        )
    if is_error is not None and not isinstance(is_error, bool):
        raise SessionFormatError(f"{subject} isError must be boolean when present")
    if data.get("error") is not None and is_error is not True:
        raise SessionFormatError(
            f"{subject} carries error metadata for a non-error tool result"
        )


def assert_v4_fork_result(row: Any) -> None:
    """Validate fork-generated not-started results without changing identity."""
    if not is_json_object(row) or row.get("type") != "tool/result":
        return
    data = row.get("data")
    if not is_json_object(data):
        return
    error = data.get("error")
    message = data.get("message")
    if (
        not is_json_object(error)
        or error.get("code") != "TOOL_NOT_STARTED"
        or not is_json_object(message)
        or not isinstance(message.get("id"), str)
        or not message["id"].startswith("forked-tool-result-")
    ):
        return
    source = message.get("source")
    call_id = source.get("callId") if is_json_object(source) else None
    prefix = f"forked-tool-result-{call_id}-"
    suffix = message["id"][len(prefix) :]
    operation = row.get("surfaceOp")
    replacement = is_json_object(operation) and operation.get("op") == "replace"
    try:
        sequence = int(suffix)
    except (TypeError, ValueError):
        sequence = -1
    content = message.get("content")
    text = content[0] if isinstance(content, list) and len(content) == 1 else None
    source_event_seqs = row.get("sourceEventSeqs")
    valid = (
        isinstance(call_id, str)
        and message["id"].startswith(prefix)
        and len(suffix) > 0
        and suffix.isdigit()
        and (suffix == "0" or not suffix.startswith("0"))
        and _is_safe_integer(sequence)
        and (
            (replacement and isinstance(row.get("seq"), int) and sequence < row["seq"])
            or (not replacement and sequence == row.get("seq"))
        )
        and error.get("name") == "ToolNotStartedError"
        and (
            (not replacement and source_event_seqs is None and operation == "append")
            or (
                replacement
                and isinstance(source_event_seqs, list)
                and len(source_event_seqs) == 1
                and source_event_seqs[0] == sequence
            )
        )
        and message.get("role") == "tool"
        and message.get("isError") is True
        and message.get("toolCallId") == call_id
        and is_json_object(source)
        and source.get("kind") == "tool"
        and is_json_object(text)
        and text.get("type") == "text"
        and isinstance(text.get("text"), str)
    )
    if not valid:
        raise SessionFormatError("invalid V4 not-started fork result")


def assert_v4_message_sources(event: JsonObject) -> None:
    """Validate producer attribution in every declared durable message slot."""

    def check(message: JsonObject) -> JsonObject:
        source = message.get("source")
        if (
            not is_json_object(source)
            or not isinstance(source.get("kind"), str)
            or not source["kind"]
            or source["kind"] == "plugin"
        ):
            raise SessionFormatError(
                "format v4 message requires a producer-owned source kind"
            )
        return message

    _map_event_messages(event, check)


def assert_v4_source_row_admission(row: Any) -> None:
    """Refuse retired source syntax even after a malformed recoverable row."""
    if not is_json_object(row) or not is_json_object(row.get("data")):
        return
    data = row["data"]
    row_type = row.get("type")
    if row_type == "user/message":
        messages = [data]
    elif row_type in ("system/message", "assistant/message", "tool/result"):
        messages = [data.get("message")]
    elif row_type == "agent/inbox/spliced":
        messages = data.get("inserted")
    elif row_type == "session/title-llm-request":
        messages = data.get("messages")
    else:
        messages = []
    if not isinstance(messages, list):
        return
    for message in messages:
        if not is_json_object(message):
            continue
        source = message.get("source")
        if is_json_object(source) and source.get("kind") == "plugin":
            raise SessionFormatError(
                "format v4 message requires a producer-owned source kind"
            )


def assert_v4_request_header_data(row: Any) -> None:
    """Validate native v4 request/header payloads at load time.

    The same predicate set the append layer enforces, except for the
    one legacy form the migration itself emits: a header carried as
    ``{"header": {}, "reason": {"kind": "start"}}``. That pair is
    grandfathered; any other shape must carry a config with a nonempty
    provider/model and a reason from the current taxonomy.
    """
    if not is_json_object(row) or row.get("type") != "request/header":
        return
    data = row.get("data")
    if not is_json_object(data):
        return
    header = data.get("header")
    if not is_json_object(header):
        return
    if header == {} and data.get("reason") == {"kind": "start"}:
        return
    subject = f"format v4 request/header at seq {row.get('seq')}"
    config = header.get("config")
    if (
        not is_json_object(config)
        or not isinstance(config.get("provider"), str)
        or not config["provider"]
        or not isinstance(config.get("model"), str)
        or not config["model"]
    ):
        raise SessionFormatError(
            f"{subject} requires a config with a nonempty provider and model"
        )
    reasoning_effort = config.get("reasoningEffort")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or not reasoning_effort
    ):
        raise SessionFormatError(
            f"{subject} config reasoningEffort must be a nonempty string"
        )
    if data.get("reason") not in ("initial", "resume", "change", "series"):
        raise SessionFormatError(
            f'{subject} reason must be one of "initial", "resume", "change", "series"'
        )
    if "startsSeries" in data and data["startsSeries"] is not True:
        raise SessionFormatError(f"{subject} startsSeries must be true when present")
    defaults = header.get("adapterDefaults")
    if defaults is not None:
        if not is_json_object(defaults):
            raise SessionFormatError(f"{subject} adapterDefaults must be an object")
        for key in defaults:
            if key not in ("reasoningEffort", "maxTokens"):
                raise SessionFormatError(
                    f"{subject} adapterDefaults carries unknown key {key!r}"
                )
        for marker in ("reasoningEffort", "maxTokens"):
            if marker in defaults and defaults[marker] is not True:
                raise SessionFormatError(
                    f"{subject} adapterDefaults {marker} must be true when present"
                )
            if defaults.get(marker) is True and marker not in config:
                raise SessionFormatError(
                    f"{subject} adapterDefaults {marker} requires config {marker}"
                )
    tools = header.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            raise SessionFormatError(f"{subject} header tools must be an array")
        for tool in tools:
            if not is_json_object(tool):
                raise SessionFormatError(f"{subject} header tools must be objects")
            if "deferLoading" in tool and tool["deferLoading"] is not True:
                raise SessionFormatError(
                    f"{subject} header tool deferLoading must be true when present"
                )


def assert_v4_request_context_data(row: Any) -> None:
    """Validate native v4 request/context payloads at load time."""
    if not is_json_object(row) or row.get("type") != "request/context":
        return
    data = row.get("data")
    if not is_json_object(data):
        return
    subject = f"format v4 request/context at seq {row.get('seq')}"
    if (
        not isinstance(data.get("provider"), str)
        or not data["provider"]
        or not isinstance(data.get("model"), str)
        or not data["model"]
    ):
        raise SessionFormatError(f"{subject} requires a nonempty provider and model")
    context_window = data.get("contextWindow")
    if context_window is not None and (
        not _is_safe_integer(context_window) or context_window <= 0
    ):
        raise SessionFormatError(
            f"{subject} contextWindow must be a positive safe integer when present"
        )
    if "systemPromptUpdate" in data and data["systemPromptUpdate"] != "in-history":
        raise SessionFormatError(
            f'{subject} systemPromptUpdate must be "in-history" when present'
        )


def assert_v4_assistant_settlement_data(row: Any) -> None:
    """Validate assistant settlement payloads at load time.

    Both settlement types require non-negative turn/step and a list
    stream. An assistant message additionally requires an identified
    model message. Provider/model attribution is required when the
    source carries either key; a source carrying neither is
    grandfathered, because migrated logs predate provider/model
    attribution on assistant sources.
    """
    if not is_json_object(row) or row.get("type") not in (
        "assistant/message",
        "assistant/attempt",
    ):
        return
    data = row.get("data")
    if not is_json_object(data):
        return
    subject = f"format v4 {row.get('type')} at seq {row.get('seq')}"
    for count_field in ("turn", "step"):
        require_count(data.get(count_field), f"{subject} {count_field}")
    if not isinstance(data.get("stream"), list):
        raise SessionFormatError(f"{subject} stream must be an array")
    if row.get("type") != "assistant/message":
        return
    if "tool_calls" in data:
        raise SessionFormatError(f"{subject} must not carry tool_calls")
    if "interrupted" in data and data["interrupted"] is not True:
        raise SessionFormatError(f"{subject} interrupted must be true when present")
    message = data.get("message")
    if not is_json_object(message):
        raise SessionFormatError(f"{subject} requires a message object")
    source = message.get("source")
    if (
        not isinstance(message.get("id"), str)
        or not message["id"]
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), list)
        or not is_json_object(source)
        or source.get("kind") != "model"
    ):
        raise SessionFormatError(f"{subject} requires an identified model message")
    if ("provider" in source or "model" in source) and (
        not isinstance(source.get("provider"), str)
        or not source["provider"]
        or not isinstance(source.get("model"), str)
        or not source["model"]
    ):
        raise SessionFormatError(f"{subject} requires a nonempty provider and model")


def assert_v4_row_admission(
    row: Any, known_event_types: frozenset[str] | None = None
) -> None:
    """Apply native v4 admission before a scanner discards a recoverable tail."""
    if is_json_object(row):
        if (
            row.get("type") == "developer/message"
            and row.get("ignorable") is True
            and (
                known_event_types is None
                or "developer/message" not in known_event_types
            )
        ):
            return
        assert_v4_developer_data(row)
    assert_v4_source_row_admission(row)
    assert_v4_retired_syntax(row)
    assert_v4_request_header_data(row)
    assert_v4_request_context_data(row)
    assert_v4_assistant_settlement_data(row)
    assert_v4_system_message_fields(row)
    if not is_json_object(row) or row.get("type") != "tool/result":
        return
    assert_v4_tool_result_message(row)
    assert_v4_fork_result(row)


# ---------------------------------------------------------------------------
# v3 codec (v2 framing + v3 admission) and v4 codec (v2 framing, version 4)
# ---------------------------------------------------------------------------


def assert_v3_event_admission(event: JsonObject) -> None:
    """Refuse required predecessor PTC tags without interpreting payloads."""
    if event.get("type") in ("tool/code-dispatch-start", "tool/code-dispatch") and (
        event.get("ignorable") is not True
    ):
        raise SessionFormatUnsupportedMigrationError(
            "format v3 contains unknown event type "
            f"{json.dumps(event['type'])} at seq {event.get('seq')}"
        )


def _assert_v3_system(data: JsonObject) -> None:
    exact_keys(data, ["turn", "step", "message"], [], "system/message data")
    for field_name in ("turn", "step"):
        if require_count(data.get(field_name), field_name) == 0:
            raise SessionFormatError(f"{field_name} must be positive")
    message = require_record(data.get("message"), "system message")
    exact_keys(message, ["id", "role", "source", "content"], [], "system message")
    if (
        not isinstance(message.get("id"), str)
        or not message["id"]
        or message.get("role") != "system"
    ):
        raise SessionFormatError("system message requires an id and system role")
    source = require_record(message.get("source"), "system source")
    if (
        source.get("kind") != "plugin"
        or not isinstance(source.get("plugin"), str)
        or not source["plugin"]
    ):
        raise SessionFormatError("system message requires plugin source")


def assert_v3_structural_row(value: Any) -> None:
    """Reject v3 structural payload violations beyond recoverable row failure."""
    if not is_json_object(value):
        return
    if value.get("type") == "request/header":
        data = require_record(value.get("data"), "request/header data")
        if "system" in require_record(data.get("header"), "request header"):
            raise SessionFormatUnsupportedMigrationError(
                "format v3 request/header rejects retired header.system"
            )
    elif value.get("type") == "system/message":
        data = require_record(value.get("data"), "system/message data")
        _assert_v3_system(data)


def assert_v3_event(
    event: JsonObject, known_event_types: frozenset[str] | None = None
) -> None:
    """Validate one canonical v3 event without interpreting plugin payloads."""
    record = require_record(event, "format v3 event")
    subject = f"format v3 {event.get('type')} at seq {event.get('seq')}"
    obsolete = event.get("type") in ("tool/code-dispatch-start", "tool/code-dispatch")
    known = not obsolete and (
        event.get("type") in SURFACE_TYPES
        or event.get("type") in RELEASED_V2_EVENT_DISPOSITIONS
        or event.get("type") in ("tool/ptc-dispatch-start", "tool/ptc-dispatch")
        or event.get("type") in ("feedback/message-put", "feedback/message-delete")
        or (known_event_types is not None and event.get("type") in known_event_types)
    )
    opaque = not known
    exact_keys(
        record,
        ["type", "seq", "time", "data"],
        (
            ["ignorable", "surfaceOp", "sourceEventSeqs"]
            if event.get("type") in SURFACE_TYPES or opaque
            else ["ignorable"]
        ),
        subject,
    )
    if not isinstance(event.get("type"), str):
        raise SessionFormatError(f"{subject} type must be a string")
    require_count(event.get("seq"), f"{subject} seq")
    require_safe_integer(event.get("time"), f"{subject} time")
    if "ignorable" in record and record["ignorable"] is not True:
        raise SessionFormatError(f"{subject} ignorable must be true when present")
    if event.get("type") in SURFACE_TYPES:
        operation = record.get("surfaceOp")
        if operation is None:
            raise SessionFormatError(f"{subject} requires a surfaceOp marker")
        if operation != "append":
            replace = require_record(operation, f"{subject} surfaceOp")
            if (
                len(replace) != 3
                or replace.get("op") != "replace"
                or "startSeq" not in replace
                or "endSeq" not in replace
            ):
                raise SessionFormatError(
                    f"{subject} requires exact replace fields op/startSeq/endSeq"
                )
            for key in ("startSeq", "endSeq"):
                if (
                    require_count(replace[key], f"{subject} surfaceOp {key}")
                    >= event["seq"]
                ):
                    raise SessionFormatError(
                        f"{subject} replacement endpoints must reference earlier events"
                    )
        sources = record.get("sourceEventSeqs")
        if event.get("type") == "assistant/message" and sources is not None:
            raise SessionFormatError(
                f"{subject} embeds its stream and cannot carry sourceEventSeqs"
            )
        if sources is not None:
            if not isinstance(sources, list) or not sources:
                raise SessionFormatError(
                    f"{subject} sourceEventSeqs must be a non-empty array"
                )
            seen: set[int] = set()
            for source in sources:
                seq = require_count(source, f"{subject} sourceEventSeqs member")
                if seq >= event["seq"] or seq in seen:
                    raise SessionFormatError(
                        f"{subject} sourceEventSeqs must be unique earlier seqs"
                    )
                seen.add(seq)
    assert_v3_structural_row(event)
    if event.get("type") == "request/header":
        data = require_record(event.get("data"), f"{subject} data")
        header = require_record(data.get("header"), f"{subject} header")
        if (isinstance(header.get("tools"), list) and not header["tools"]) or (
            is_json_object(header.get("adapterDefaults"))
            and not header["adapterDefaults"]
        ):
            raise SessionFormatError(
                f"{subject} empty optional header fields must be omitted"
            )
    if event.get("type") != "tool/result":
        return
    data = require_record(event.get("data"), f"{subject} data")
    if data.get("error") is None:
        return
    message = require_record(data.get("message"), f"{subject} message")
    content = message.get("content")
    if (
        not isinstance(content, list)
        or len(content) != 1
        or not is_json_object(content[0])
        or content[0].get("type") != "tool-result"
        or content[0].get("isError") is not True
    ):
        raise SessionFormatError(
            f"{subject} carries error metadata for a non-error tool result"
        )


class _V3Decoder(_Decoder):
    """v3 decoder: v2 framing plus structural and envelope admission."""

    def __init__(self, header_value: Any, recovery: str) -> None:
        physical = require_record(header_value, "format v3 physical header")
        if physical.get("version") != 3:
            raise SessionFormatError("expected format v3 physical Session header")
        self._inner = _V2Decoder({**physical, "version": 2}, recovery)
        self.header = {**self._inner.header, "version": 3}
        self.header_inherited_event_count = None
        self._recoverable = recovery == "recoverable"
        self._issue: SessionFormatError | None = None
        self._accepted_cut: int | None = None

    def decode_row(self, row: Any, emit: _Emit) -> None:
        assert_v3_structural_row(row)
        if is_json_object(row):
            assert_v3_event_admission(row)

        outer = self

        class _V3Emit(_Emit):
            def emit_event(self, event: JsonObject) -> None:
                assert_v3_event_admission(event)
                if outer._issue is None:
                    try:
                        assert_v3_event(event)
                    except SessionFormatError as error:
                        if not outer._recoverable:
                            raise
                        outer._issue = error
                if outer._issue is not None:
                    if event["type"] == "turn/end":
                        raise outer._issue
                    return
                if (
                    event["type"] == "session/end-seed"
                    and is_json_object(event.get("data"))
                    and event["data"].get("inherited") is True
                ):
                    outer._accepted_cut = event["seq"]
                emit.emit_event(event)

            # pragma: no cover - unreachable: the inner v2 decoder only ever
            # calls emit.emit_event, never emit_run; _V3Emit is a closure
            # class defined inside decode_row, so it cannot be driven directly.
            def emit_run(self, run: SessionFormatEventRun) -> None:  # pragma: no cover
                emit.emit_run(run)

        self._inner.decode_row(row, _V3Emit())

    def finish(self) -> int:
        if self._issue is None:
            return self._inner.finish()
        if self.header["isSeeded"] and self._accepted_cut is None:
            raise SessionFormatError(
                "format v3 seeded Session lacks an accepted inherited end-seed marker"
            )
        if not self.header["isSeeded"] and self._accepted_cut is not None:
            raise SessionFormatError(
                "format v3 unseeded Session contains an inherited end-seed marker"
            )
        return self._accepted_cut if self._accepted_cut is not None else 0


class _V3Codec:
    """Frozen physical JSON codec for released v3."""

    version = 3

    def decode_header(self, value: Any) -> JsonObject:
        physical = require_record(value, "format v3 physical header")
        if physical.get("version") != 3:
            raise SessionFormatError("expected format v3 physical Session header")
        return {
            **_V2Decoder._decode_physical_header({**physical, "version": 2}),
            "version": 3,
        }

    def create_decoder(self, header_value: Any, recovery: str) -> _Decoder:
        return _V3Decoder(header_value, recovery)

    def encode_header(
        self, header: JsonObject, inherited_event_count: int
    ) -> JsonObject:
        assert_v3_header(header)
        return {
            **_encode_v2_header({**header, "version": 2}, inherited_event_count),
            "version": 3,
        }

    def encode_event(self, event: JsonObject) -> JsonObject:
        assert_v3_event_admission(event)
        assert_v3_event(event)
        return _V2Codec().encode_event(event)


class _V4Decoder(_Decoder):
    """v4 decoder: v2 framing (version rewritten) plus native v4 admission."""

    def __init__(self, header_value: Any, recovery: str) -> None:
        physical = require_record(header_value, "format v4 physical header")
        if physical.get("version") != 4:
            raise SessionFormatError("expected format v4 physical header")
        self._inner = _V2Decoder({**physical, "version": 2}, recovery)
        self.header = {**self._inner.header, "version": 4}
        self.header_inherited_event_count = None

    def decode_row(self, row: Any, emit: _Emit) -> None:
        assert_v4_row_admission(row)
        self._inner.decode_row(row, emit)

    def finish(self) -> int:
        return self._inner.finish()


class _V4Codec:
    """Frozen physical JSON codec for released v4 (current encoder)."""

    version = 4

    def decode_header(self, value: Any) -> JsonObject:
        physical = require_record(value, "format v4 physical header")
        if physical.get("version") != 4:
            raise SessionFormatError("expected format v4 physical header")
        return {
            **_V2Decoder._decode_physical_header({**physical, "version": 2}),
            "version": 4,
        }

    def create_decoder(self, header_value: Any, recovery: str) -> _Decoder:
        return _V4Decoder(header_value, recovery)

    def encode_header(
        self, header: JsonObject, inherited_event_count: int
    ) -> JsonObject:
        assert_v4_header(header)
        return {
            **_encode_v2_header({**header, "version": 2}, inherited_event_count),
            "version": 4,
        }

    def encode_event(self, event: JsonObject) -> JsonObject:
        if event.get("type") == "developer/message" and event.get("ignorable") is True:
            assert_v4_developer_data(event)
            assert_v4_retired_syntax(event)
        assert_v4_row_admission(event)
        return _V2Codec().encode_event(event)


# ---------------------------------------------------------------------------
# Assistant stream accumulator (compact attempt streams)
# ---------------------------------------------------------------------------


def _safe_gap(previous_time: int, time: int) -> int | None:
    gap = time - previous_time
    if not _is_safe_integer(gap):
        return None
    return gap


class AssistantStreamAccumulator:
    """Compact timed model chunks into durable attempt stream records.

    Adjacent deltas of the same kind and index merge into one packed record
    (``text-chunks`` / ``reasoning-chunks`` / ``tool-call-chunks``) with
    per-member time gaps; block markers, usage, and finish records stay raw.
    """

    def __init__(self) -> None:
        self._records: list[JsonObject] = []

    def push(self, time: int, chunk: JsonObject) -> None:
        safe_time = require_safe_integer(time, "chunk time")
        chunk_type = chunk.get("type")
        previous = self._records[-1] if self._records else None
        if chunk_type in ("text-delta", "reasoning-delta"):
            if not isinstance(chunk.get("index"), int):
                raise SessionFormatError(f"{chunk_type} index must be an integer")
            if not isinstance(chunk.get("text"), str):
                raise SessionFormatError(f"{chunk_type} text must be a string")
            record_type = (
                "text-chunks" if chunk_type == "text-delta" else "reasoning-chunks"
            )
            gap = (
                _safe_gap(previous["last_time"], safe_time)
                if previous is not None and previous["type"] == record_type
                else None
            )
            if (
                previous is not None
                and previous["type"] == record_type
                and previous["index"] == chunk["index"]
                and gap is not None
            ):
                previous["dt"].append(gap)
                previous["texts"].append(chunk["text"])
                previous["last_time"] = safe_time
            else:
                self._records.append(
                    {
                        "type": record_type,
                        "time0": safe_time,
                        "index": chunk["index"],
                        "dt": [],
                        "texts": [chunk["text"]],
                        "last_time": safe_time,
                    }
                )
            return
        if chunk_type == "tool-call-delta":
            if not isinstance(chunk.get("index"), int):
                raise SessionFormatError("tool-call-delta index must be an integer")
            if not isinstance(chunk.get("id"), str):
                raise SessionFormatError("tool-call-delta id must be a string")
            if "name" in chunk and not isinstance(chunk["name"], str):
                raise SessionFormatError("tool-call-delta name must be a string")
            if not isinstance(chunk.get("argumentsDelta"), str):
                raise SessionFormatError(
                    "tool-call-delta argumentsDelta must be a string"
                )
            if not chunk["id"] or chunk.get("name") == "":
                self._records.append(
                    {"type": "chunk", "time": safe_time, "chunk": chunk}
                )
                return
            gap = (
                _safe_gap(previous["last_time"], safe_time)
                if previous is not None and previous["type"] == "tool-call-chunks"
                else None
            )
            same_name = (
                previous is not None
                and previous["type"] == "tool-call-chunks"
                and ("name" in previous) == ("name" in chunk)
                and previous.get("name") == chunk.get("name")
            )
            if (
                previous is not None
                and previous["type"] == "tool-call-chunks"
                and previous["index"] == chunk["index"]
                and previous["id"] == chunk["id"]
                and same_name
                and gap is not None
            ):
                previous["dt"].append(gap)
                previous["args"].append(chunk["argumentsDelta"])
                previous["last_time"] = safe_time
            else:
                self._records.append(
                    {
                        "type": "tool-call-chunks",
                        "time0": safe_time,
                        "index": chunk["index"],
                        "dt": [],
                        "id": chunk["id"],
                        **({"name": chunk["name"]} if "name" in chunk else {}),
                        "args": [chunk["argumentsDelta"]],
                        "last_time": safe_time,
                    }
                )
            return
        if chunk_type in ("block-start", "block-end", "usage", "finish"):
            self._records.append({"type": "chunk", "time": safe_time, "chunk": chunk})
            return
        raise SessionFormatError(f"unknown stream chunk type {chunk_type!r}")

    def snapshot(self) -> list[JsonObject]:
        """Return the current compact attempt stream (durable records only)."""
        records: list[JsonObject] = []
        for record in self._records:
            if record["type"] == "chunk":
                records.append(dict(record))
                continue
            durable = {k: v for k, v in record.items() if k != "last_time"}
            if record["type"] == "tool-call-chunks":
                durable = {
                    **durable,
                    "dt": list(durable["dt"]),
                    "args": list(durable["args"]),
                }
            else:
                durable = {
                    **durable,
                    "dt": list(durable["dt"]),
                    "texts": list(durable["texts"]),
                }
            records.append(durable)
        return records


# ---------------------------------------------------------------------------
# Migration declarations, stages, chain
# ---------------------------------------------------------------------------


@dataclass
class SessionFormatMigrationStageInput:
    """Headers and inherited cut supplied when one adjacent body stage is created."""

    source_header: JsonObject
    target_header: JsonObject
    source_inherited_event_count: int | None
    source_kind: str  # 'decoded' | 'transformed'


@dataclass
class SessionFormatMigration:
    """One independently maintained adjacent streaming migration."""

    name: str
    from_version: int
    to_version: int
    migrate_header: Callable[[JsonObject], JsonObject]
    create_stage: Callable[[SessionFormatMigrationStageInput], _Stage]
    validate_target_header: Callable[[Any], JsonObject]


class _Stage:
    """Stateful adjacent migration stage used by streaming restores."""

    header_inherited_event_count: int | None = None

    def transform_event(self, event: JsonObject, emit: _Emit) -> None:
        """Transform one source event, emitting every settled target item."""

    def transform_run(self, run: SessionFormatEventRun, emit: _Emit) -> None:
        """Transform one compact source run without expanding it first."""
        for event in run.expand():
            self.transform_event(event, emit)

    def finish(self, emit: _Emit) -> int:
        """Emit trailing target items and return the exact target cut."""
        return 0


def define_session_format_migration(
    migration: SessionFormatMigration,
) -> SessionFormatMigration:
    """Validate and freeze one adjacent migration declaration."""
    if not migration.name:
        raise SessionFormatError("Session migration name must be a non-empty string")
    from_version = require_format_version(
        migration.from_version, f"{migration.name} fromVersion"
    )
    to_version = require_format_version(
        migration.to_version, f"{migration.name} toVersion"
    )
    if to_version != from_version + 1:
        raise SessionFormatError(
            f"{migration.name} must declare adjacent v{from_version}->v{from_version + 1}"
        )
    return migration


def _refusal(
    migration: SessionFormatMigration, error: BaseException, subject: str = "Session"
) -> SessionFormatUnsupportedMigrationError:
    if isinstance(error, SessionFormatUnsupportedMigrationError):
        return error
    detail = str(error)
    return SessionFormatUnsupportedMigrationError(
        f"{migration.name} refuses this format v{migration.from_version} "
        f"{subject}: {detail}"
    )


class _ChainedContext(_Emit):
    def __init__(
        self, migration: SessionFormatMigration, stage: _Stage, downstream: _Emit
    ) -> None:
        self._migration = migration
        self._stage = stage
        self._downstream = downstream

    def emit_event(self, event: JsonObject) -> None:
        try:
            self._stage.transform_event(event, self._downstream)
        except SessionFormatError as error:
            raise _refusal(self._migration, error) from error

    def emit_run(self, run: SessionFormatEventRun) -> None:
        try:
            self._stage.transform_run(run, self._downstream)
        except SessionFormatError as error:
            raise _refusal(self._migration, error) from error

    def finish(self) -> int:
        try:
            target_cut = self._stage.finish(self._downstream)
        except SessionFormatError as error:
            raise _refusal(self._migration, error) from error
        declared = self._stage.header_inherited_event_count
        if declared is not None and declared != target_cut:
            raise SessionFormatError(
                f"{self._migration.name} changed its predeclared inherited cut"
            )
        return target_cut


class _MigrationStream(_Emit):
    def __init__(
        self,
        header: JsonObject,
        source_inherited_event_count: int | None,
        entries: list[tuple[SessionFormatMigration, _Stage]],
        output: _Emit,
    ) -> None:
        self.header = header
        self._source_cut = source_inherited_event_count
        downstream: _Emit = output
        chained: list[_ChainedContext] = []
        for migration, stage in reversed(entries):
            context = _ChainedContext(migration, stage, downstream)
            chained.insert(0, context)
            downstream = context
        self._input = downstream
        self._chained = chained

    def emit_event(self, event: JsonObject) -> None:
        self._input.emit_event(event)

    def emit_run(self, run: SessionFormatEventRun) -> None:
        self._input.emit_run(run)

    def finish(self) -> int:
        inherited: int | None = self._source_cut
        for stage in self._chained:
            inherited = stage.finish()
        return require_count(inherited, "finished Session inherited event count")


class SessionFormatChain:
    """Pure adjacent planner and streaming migration compiler."""

    def __init__(
        self,
        current_version: int,
        migrations: list[SessionFormatMigration],
        restore_current_header: Callable[[JsonObject], JsonObject],
    ) -> None:
        self.current_version = require_format_version(
            current_version, "current Session format version"
        )
        self._restore_current_header = restore_current_header
        by_from: dict[int, SessionFormatMigration] = {}
        names: set[str] = set()
        for candidate in migrations:
            migration = define_session_format_migration(candidate)
            if migration.from_version in by_from:
                raise SessionFormatError(
                    f"Session migration v{migration.from_version}->"
                    f"v{migration.to_version} is duplicated"
                )
            if migration.name in names:
                raise SessionFormatError(
                    f"Session migration name {json.dumps(migration.name)} is duplicated"
                )
            by_from[migration.from_version] = migration
            names.add(migration.name)
        ordered: list[SessionFormatMigration] = []
        for version in range(self.current_version):
            planned = by_from.get(version)
            if planned is None:
                raise SessionFormatUnsupportedMigrationError(
                    f"Session migration v{version}->v{version + 1} is missing"
                )
            ordered.append(planned)
        if len(by_from) != len(ordered):
            invalid = next(
                version for version in by_from if version >= self.current_version
            )
            raise SessionFormatError(
                "Session migration from v"
                f"{invalid} does not lead to current v{self.current_version}"
            )
        self._migrations = ordered

    def _plan(self, from_version: int) -> list[SessionFormatMigration]:
        version = require_format_version(from_version, "stored Session format version")
        if version > self.current_version:
            raise SessionFormatUnsupportedMigrationError(
                f"stored Session uses newer format v{version}; this build writes "
                f"v{self.current_version}"
            )
        return self._migrations[version:]

    def migrate_header(self, source: JsonObject) -> JsonObject:
        """Convert only a supported header to the current logical form."""
        current = snapshot_header(source, "stored Session header")
        for migration in self._plan(current["version"]):
            current = self._advance_header(migration, current)
        current = snapshot_header(
            self._restore_current_header(current),
            "current Session header restoration",
        )
        if current["version"] != self.current_version:
            raise SessionFormatError(
                "current Session header restorer returned v"
                f"{current['version']}; expected v{self.current_version}"
            )
        return current

    def _advance_header(
        self, migration: SessionFormatMigration, source: JsonObject
    ) -> JsonObject:
        try:
            target = migration.migrate_header(
                snapshot_header(source, f"{migration.name} header input")
            )
        except SessionFormatError as error:
            raise _refusal(migration, error, "Session header") from error
        current = snapshot_header(target, f"{migration.name} header output")
        if current["version"] != migration.to_version:
            raise SessionFormatError(
                f"{migration.name} header returned v{current['version']}; "
                f"expected v{migration.to_version}"
            )
        try:
            migration.validate_target_header(current)
        except SessionFormatError as error:
            raise _refusal(migration, error, "Session header") from error
        return current

    def create_stream(
        self,
        source_header: JsonObject,
        source_cut: int | None,
        output: _Emit,
    ) -> _MigrationStream:
        """Compile the complete migration stage chain for one source artifact."""
        header = source_header
        validated_cut = (
            None
            if source_cut is None
            else require_count(source_cut, "Session inherited event count")
        )
        inherited: int | None = validated_cut
        entries: list[tuple[SessionFormatMigration, _Stage]] = []
        plan = self._plan(header["version"])
        for index, migration in enumerate(plan):
            target_header = self._advance_header(migration, header)
            try:
                stage = migration.create_stage(
                    SessionFormatMigrationStageInput(
                        source_header=header,
                        target_header=target_header,
                        source_inherited_event_count=inherited,
                        source_kind="decoded" if index == 0 else "transformed",
                    )
                )
            except SessionFormatError as error:
                raise _refusal(migration, error) from error
            header = target_header
            entries.append((migration, stage))
            inherited = stage.header_inherited_event_count
        return _MigrationStream(header, validated_cut, entries, output)


# ---------------------------------------------------------------------------
# Migration edge v0 -> v1
# ---------------------------------------------------------------------------

_LEGACY_ASSISTANT_SOURCE_KEY = "provenance"


def _assert_header_version(header: JsonObject, version: int) -> None:
    if header.get("version") != version:
        raise SessionFormatError(f"expected format v{version} header")


def _legacy_message_id(session_id: str, seq: int) -> str:
    return f"legacy-message:{session_id}:{seq}"


def _malformed_legacy(session_id: str, event_type: str, seq: int) -> SessionFormatError:
    return SessionFormatError(
        f"session {json.dumps(session_id)} contains malformed pre-react-loop "
        f"{event_type} at seq {seq}"
    )


def _normalize_legacy_compaction_type(event: JsonObject) -> JsonObject:
    renamed = {
        "compact/start": "compaction/start",
        "compact/summary": "compaction/summary",
        "compact/end": "compaction/end",
        "compact/prune": "compaction/prune",
    }
    new_type = renamed.get(str(event.get("type")))
    if new_type is None:
        return event
    return {**event, "type": new_type}


def _assert_supported_legacy_type(event: JsonObject, session_id: str) -> None:
    if event.get("type") in ("request/header-delta", "mode/set"):
        raise SessionFormatUnsupportedMigrationError(
            f"session {json.dumps(session_id)} contains unsupported legacy "
            f"{event['type']} event at seq {event.get('seq')}"
        )
    if event.get("type") == "request/header":
        data = require_record(
            event.get("data"), f"request/header {event.get('seq')} data"
        )
        if data.get("reason") == "fallback":
            raise SessionFormatUnsupportedMigrationError(
                f"session {json.dumps(session_id)} contains unsupported "
                f'request/header reason "fallback" at seq {event.get("seq")}'
            )


def _normalize_legacy_turn_start(event: JsonObject, session_id: str) -> JsonObject:
    if event.get("type") != "turn/start":
        return event
    data = require_record(event.get("data"), f"turn/start {event.get('seq')} data")
    if "trigger" not in data:
        return event
    exact_keys(data, ["turn", "trigger"], [], f"turn/start {event.get('seq')} data")
    turn = require_count(data["turn"], f"turn/start {event.get('seq')} turn")
    trigger = require_record(data["trigger"], f"turn/start {event.get('seq')} trigger")
    if turn < 1 or not isinstance(trigger.get("kind"), str) or not trigger["kind"]:
        raise _malformed_legacy(session_id, "turn/start", event["seq"])
    return {**event, "data": {"turn": turn}}


def _normalize_legacy_error_reason(
    reason: JsonObject, seq: int, session_id: str
) -> JsonObject:
    require_count(reason.get("step"), f"turn/end {seq} error step")
    failure = reason.get("failure")
    if failure is not None:
        exact_keys(reason, ["kind", "step", "failure"], [], f"turn/end {seq} reason")
        record = require_record(failure, f"turn/end {seq} failure")
        exact_keys(
            record,
            ["message", "code"],
            ["status", "providerRetryAfterMs", "requestId"],
            f"turn/end {seq} failure",
        )
        if not isinstance(record.get("message"), str) or not isinstance(
            record.get("code"), str
        ):
            raise _malformed_legacy(session_id, "turn/end", seq)
        return {"kind": "error", "error": record}
    exact_keys(reason, ["kind", "step", "message"], ["code"], f"turn/end {seq} reason")
    if not isinstance(reason.get("message"), str) or (
        reason.get("code") is not None and not isinstance(reason["code"], str)
    ):
        raise _malformed_legacy(session_id, "turn/end", seq)
    return {
        "kind": "error",
        "error": {
            "message": reason["message"],
            "code": reason["code"]
            if isinstance(reason.get("code"), str)
            else "UNKNOWN",
        },
    }


def _normalize_legacy_turn_end(event: JsonObject, session_id: str) -> JsonObject:
    if event.get("type") != "turn/end":
        return event
    data = require_record(event.get("data"), f"turn/end {event.get('seq')} data")
    exact_keys(data, ["turn", "reason"], [], f"turn/end {event.get('seq')} data")
    turn = require_count(data["turn"], f"turn/end {event.get('seq')} turn")
    if turn < 1:
        raise _malformed_legacy(session_id, "turn/end", event["seq"])
    reason = require_record(data["reason"], f"turn/end {event.get('seq')} reason")
    if not isinstance(reason.get("kind"), str):
        raise _malformed_legacy(session_id, "turn/end", event["seq"])
    kind = reason["kind"]
    if kind in ("completed", "blocked", "max-tokens", "interrupted"):
        exact_keys(reason, ["kind"], [], f"turn/end {event.get('seq')} reason")
        return event
    if kind == "aborted":
        if "reason" in reason:
            return event
        exact_keys(reason, ["kind"], [], f"turn/end {event.get('seq')} reason")
        current: JsonObject = {"kind": "aborted", "reason": {"kind": "legacy"}}
    elif kind == "disposed":
        exact_keys(reason, ["kind"], [], f"turn/end {event.get('seq')} reason")
        current = {"kind": "aborted", "reason": {"kind": "disposed"}}
    elif kind == "error":
        if "error" in reason:
            return event
        current = _normalize_legacy_error_reason(reason, event["seq"], session_id)
    else:
        return event
    return {**event, "data": {**data, "reason": current}}


def _normalize_legacy_request_header(event: JsonObject, session_id: str) -> JsonObject:
    if event.get("type") != "request/header":
        return event
    data = require_record(event.get("data"), f"request/header {event.get('seq')} data")
    header = require_record(data["header"], f"request/header {event.get('seq')} header")
    if "messagePrefix" not in header:
        return event
    if not isinstance(header["messagePrefix"], list):
        raise SessionFormatError(
            f"session {json.dumps(session_id)} contains malformed "
            f"request/header messagePrefix at seq {event.get('seq')}"
        )
    current_header = {k: v for k, v in header.items() if k != "messagePrefix"}
    return {**event, "data": {**data, "header": current_header}}


def _normalize_legacy_steering(event: JsonObject, session_id: str) -> JsonObject:
    if event.get("type") != "steering/message":
        return event
    data = require_record(
        event.get("data"), f"steering/message {event.get('seq')} data"
    )
    wrapped = data.get("message")
    if wrapped is not None:
        exact_keys(
            data, ["turn", "message"], [], f"steering/message {event.get('seq')} data"
        )
        require_count(data["turn"], f"steering/message {event.get('seq')} turn")
        return {**event, "type": "user/message", "data": wrapped}
    exact_keys(
        data,
        ["turn", "content", "source"],
        [],
        f"steering/message {event.get('seq')} data",
    )
    require_count(data["turn"], f"steering/message {event.get('seq')} turn")
    message = {k: v for k, v in data.items() if k != "turn"}
    return {
        **event,
        "type": "user/message",
        "data": {
            **message,
            "id": _legacy_message_id(session_id, event["seq"]),
            "role": "user",
        },
    }


def _normalize_legacy_retry(
    event: JsonObject, session_id: str, retry_ids: dict[str, str]
) -> JsonObject:
    if event.get("type") != "llm/retry":
        return event
    data = require_record(event.get("data"), f"llm/retry {event.get('seq')} data")
    chain = "\0".join(
        json.dumps(data.get(key)) for key in ("turn", "step", "provider", "policyKey")
    )
    retry_id = data.get("retryId")
    if isinstance(retry_id, str) and retry_id:
        retry_ids[chain] = retry_id
        return event
    if "retryId" in data:
        return event
    migrated = retry_ids.get(chain, f"legacy-retry:{session_id}:{event['seq']}")
    retry_ids[chain] = migrated
    return {**event, "data": {**data, "retryId": migrated}}


def _add_legacy_compaction_id(event: JsonObject, compaction_id: str) -> JsonObject:
    data = require_record(event.get("data"), f"{event['type']} {event.get('seq')} data")
    if "compactionId" in data:
        return event
    return {**event, "data": {**data, "compactionId": compaction_id}}


class _LegacyCompactionState:
    def __init__(self) -> None:
        self.message_ids: dict[int, str] = {}
        self.retry_ids: dict[str, str] = {}
        self.compaction_id: str | None = None


def _normalize_legacy_compaction(
    event: JsonObject, session_id: str, state: _LegacyCompactionState
) -> JsonObject:
    if event.get("type") == "session/end-seed":
        state.compaction_id = None
        return event
    if event.get("type") == "compaction/start":
        data = require_record(
            event.get("data"), f"compaction/start {event.get('seq')} data"
        )
        existing = data.get("compactionId")
        if isinstance(existing, str) and existing:
            state.compaction_id = existing
            return event
        if "compactionId" in data:
            return event
        new_id = f"legacy-compaction:{session_id}:{event['seq']}"
        state.compaction_id = new_id
        return {**event, "data": {**data, "compactionId": new_id}}
    compaction_id = state.compaction_id
    if compaction_id is None:
        return event
    if event.get("type") in ("compaction/summary", "compaction/end"):
        normalized = _add_legacy_compaction_id(event, compaction_id)
        if event.get("type") == "compaction/end":
            state.compaction_id = None
        return normalized
    if event.get("type") != "user/message":
        return event
    data = require_record(event.get("data"), f"user/message {event.get('seq')} data")
    source = data.get("source")
    if (
        not is_json_object(source)
        or source.get("kind") != "plugin"
        or source.get("plugin") != "compact"
        or "compactionId" in source
    ):
        return event
    return {
        **event,
        "data": {
            **data,
            "source": {**source, "compactionId": compaction_id},
        },
    }


def _replacement_start(event: JsonObject) -> int | None:
    operation = event.get("surfaceOp")
    if not is_json_object(operation) or operation.get("op") != "replace":
        return None
    return require_count(operation.get("start"), "surface replacement start")


def _event_message_id(event: JsonObject) -> str | None:
    data = require_record(event.get("data"), f"{event['type']} {event.get('seq')} data")
    message: JsonObject | None
    if event.get("type") == "user/message":
        message = data
    else:
        inner = data.get("message")
        message = inner if is_json_object(inner) else None
    if message is None:
        return None
    message_id = message.get("id")
    return message_id if isinstance(message_id, str) else None


def _normalize_legacy_message(
    event: JsonObject, session_id: str, message_ids: dict[int, str]
) -> JsonObject:
    data = require_record(event.get("data"), f"{event['type']} {event.get('seq')} data")
    if event.get("type") == "user/message":
        if (
            "id" in data
            or "role" in data
            or "message" in data
            or "content" not in data
            or "source" not in data
        ):
            return event
        return {
            **event,
            "data": {
                **data,
                "id": _legacy_message_id(session_id, event["seq"]),
                "role": "user",
            },
        }
    if event.get("type") == "assistant/message":
        if (
            "message" in data
            or "content" not in data
            or _LEGACY_ASSISTANT_SOURCE_KEY not in data
        ):
            return event
        content = data["content"]
        event_data = {k: v for k, v in data.items() if k != "content"}
        source = require_record(
            event_data.pop(_LEGACY_ASSISTANT_SOURCE_KEY),
            f"assistant/message {event.get('seq')} legacy source",
        )
        return {
            **event,
            "data": {
                **event_data,
                "message": {
                    "id": _legacy_message_id(session_id, event["seq"]),
                    "role": "assistant",
                    "content": content,
                    "source": {**source, "kind": "model"},
                },
            },
        }
    if event.get("type") == "tool/result":
        if (
            "message" in data
            or "callId" not in data
            or "content" not in data
            or "isError" not in data
        ):
            return event
        call_id = data["callId"]
        content = data["content"]
        is_error = data["isError"]
        if (
            not isinstance(call_id, str)
            or not isinstance(is_error, bool)
            or content is None
        ):
            return event
        rest = {
            k: v for k, v in data.items() if k not in ("callId", "content", "isError")
        }
        inherited_id = _replacement_start(event)
        message_id = (
            _legacy_message_id(session_id, event["seq"])
            if inherited_id is None
            else message_ids.get(inherited_id)
        )
        if message_id is None:
            raise SessionFormatError(
                f"tool/result {event.get('seq')} replacement cites a message "
                "without identity"
            )
        return {
            **event,
            "data": {
                **rest,
                "message": {
                    "id": message_id,
                    "role": "user",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": call_id,
                            "content": content,
                            "isError": is_error,
                        }
                    ],
                    "source": {"kind": "tool", "callId": call_id},
                },
            },
        }
    return event


def _assert_source_delivery_marker(
    event: JsonObject, source_header: JsonObject, source_cut: int | None
) -> None:
    if event.get("type") != "session-log-deepseek/delivery-accepted":
        return
    data = require_record(event.get("data"), f"{event['type']} {event.get('seq')} data")
    accepted_version = data.get("sessionFormatVersion", 0)
    inherited = source_header.get("parentSession") is not None and event[
        "seq"
    ] < require_count(source_cut, "format v0 inherited event count")
    if (
        accepted_version == 0
        and not inherited
        and data.get("sessionId") != source_header.get("id")
    ):
        raise SessionFormatError(
            "current-generation delivery marker names the wrong Session"
        )


def _normalize_released_v0_event(
    event: JsonObject, session_id: str, state: _LegacyCompactionState
) -> JsonObject:
    named = _normalize_legacy_compaction_type(event)
    _assert_supported_legacy_type(named, session_id)
    start = _normalize_legacy_turn_start(named, session_id)
    end = _normalize_legacy_turn_end(start, session_id)
    header = _normalize_legacy_request_header(end, session_id)
    steering = _normalize_legacy_steering(header, session_id)
    retry = _normalize_legacy_retry(steering, session_id, state.retry_ids)
    compaction = _normalize_legacy_compaction(retry, session_id, state)
    message = _normalize_legacy_message(compaction, session_id, state.message_ids)
    if message.get("type") != "assistant/chunk":
        assert_released_event_payload(message, 0)
    message_id = _event_message_id(message)
    if message_id is not None:
        state.message_ids[message["seq"]] = message_id
    return message


class _V0ToV1Stage(_Stage):
    """Identity format edge that promotes released v0 into released v1."""

    def __init__(self, input_: SessionFormatMigrationStageInput) -> None:
        _assert_header_version(input_.source_header, 0)
        self._input = input_
        self.header_inherited_event_count = require_count(
            input_.source_inherited_event_count, "format v0 inherited event count"
        )
        self._state = _LegacyCompactionState()

    def transform_event(self, event: JsonObject, emit: _Emit) -> None:
        normalized = _normalize_released_v0_event(
            event, self._input.source_header["id"], self._state
        )
        _assert_source_delivery_marker(
            normalized,
            self._input.source_header,
            self._input.source_inherited_event_count,
        )
        emit.emit_event(normalized)

    def transform_run(self, run: SessionFormatEventRun, emit: _Emit) -> None:
        if isinstance(run, PackedAssistantChunkRun):
            emit.emit_run(run)
            return
        super().transform_run(run, emit)

    def finish(self, emit: _Emit) -> int:
        return self.header_inherited_event_count  # type: ignore[return-value]


def _v0_to_v1_migrate_header(header: JsonObject) -> JsonObject:
    _assert_header_version(header, 0)
    return {**header, "version": 1}


SESSION_FORMAT_V0_TO_V1 = SessionFormatMigration(
    name="session-format-v0-to-v1",
    from_version=0,
    to_version=1,
    migrate_header=_v0_to_v1_migrate_header,
    create_stage=lambda input_: _V0ToV1Stage(input_),
    validate_target_header=assert_v1_header,
)

# ---------------------------------------------------------------------------
# Migration edge v1 -> v2
# ---------------------------------------------------------------------------

_CHUNK_EVENT_REQUIRED = ("type", "seq", "time", "data")
_CHUNK_EVENT_OPTIONAL = ("ignorable", "sourceEventSeqs", "surfaceOp")
_CHUNK_EVENT_KEYS = frozenset((*_CHUNK_EVENT_REQUIRED, *_CHUNK_EVENT_OPTIONAL))


def _v1_v2_refusal(message: str) -> SessionFormatUnsupportedMigrationError:
    return SessionFormatUnsupportedMigrationError(message)


def _record_last_time(record: JsonObject) -> int:
    if record["type"] == "chunk":
        return require_safe_integer(record["time"], "chunk time")
    total = require_safe_integer(record["time0"], "packed run start time")
    for gap in record["dt"]:
        total += require_safe_integer(gap, "packed run time gap")
    return total


class _AttemptGroup:
    def __init__(self, turn: int, step: int) -> None:
        self.turn = turn
        self.step = step
        self.spans: list[dict[str, int]] = []
        self.stream: list[dict[str, Any]] = []
        self.accumulator: AssistantStreamAccumulator | None = None
        self.chunk_count = 0
        self.last_chunk_seq: int | None = None
        self.last_chunk_time: int | None = None
        self.terminal = False


class _LegacyTurnState:
    def __init__(self) -> None:
        self.open_turn: int | None = None
        self.open_step: int | None = None
        self.previous: JsonObject | None = None


class _V1ToV2State:
    def __init__(self, source_header: JsonObject, source_cut: int) -> None:
        self.source_header = source_header
        self.source_cut = source_cut
        self.mapping: dict[int, int] = {}
        self.legacy_turns = _LegacyTurnState()
        self.pending: dict[str, Any] | None = None
        self.target_seq = 0
        self.target_cut: int | None = None if source_header["isSeeded"] else 0
        self.last_time = source_header["createdAt"]


def _append_stream_record(
    group: _AttemptGroup, source: JsonObject, last_time: int
) -> None:
    previous = group.stream[-1] if group.stream else None
    if (
        previous is None
        or source["type"] == "chunk"
        or previous["record"]["type"] != source["type"]
    ):
        group.stream.append({"record": dict(source), "last_time": last_time})
        return
    gap = _safe_gap(previous["last_time"], source["time0"])
    if previous["record"]["index"] != source["index"] or gap is None:
        group.stream.append({"record": dict(source), "last_time": last_time})
        return
    if source["type"] == "tool-call-chunks":
        target = previous["record"]
        if target.get("id") != source.get("id") or target.get("name") != source.get(
            "name"
        ):
            group.stream.append({"record": dict(source), "last_time": last_time})
            return
        target["dt"].append(gap)
        target["dt"].extend(source["dt"])
        target["args"].extend(source["args"])
    else:
        target = previous["record"]
        target["dt"].append(gap)
        target["dt"].extend(source["dt"])
        target["texts"].extend(source["texts"])
    previous["last_time"] = last_time


def _flush_accumulator(group: _AttemptGroup) -> None:
    accumulator = group.accumulator
    if accumulator is None:
        return
    for record in accumulator.snapshot():
        _append_stream_record(group, record, _record_last_time(record))
    group.accumulator = None


def _stream_of(group: _AttemptGroup) -> list[JsonObject]:
    _flush_accumulator(group)
    return [entry["record"] for entry in group.stream]


def _record_chunk_span(
    group: _AttemptGroup, first_seq: int, event_count: int, last_time: int
) -> None:
    previous = group.spans[-1] if group.spans else None
    if (
        previous is not None
        and previous["first_seq"] + previous["event_count"] == first_seq
    ):
        previous["event_count"] += event_count
    else:
        group.spans.append({"first_seq": first_seq, "event_count": event_count})
    group.chunk_count += event_count
    group.last_chunk_seq = first_seq + event_count - 1
    group.last_chunk_time = last_time


def _matches_chunk_sources(group: _AttemptGroup, sources: list[Any]) -> bool:
    if len(sources) != group.chunk_count:
        return False
    index = 0
    for span in group.spans:
        for offset in range(span["event_count"]):
            if sources[index] != span["first_seq"] + offset:
                return False
            index += 1
    return True


def _map_one(value: Any, mapping: dict[int, int], label: str) -> int:
    source = require_count(value, label)
    mapped = mapping.get(source)
    if mapped is None:
        raise _v1_v2_refusal(f"{label} targets consumed assistant/chunk {source}")
    return mapped


def _map_list(values: list[Any], mapping: dict[int, int], label: str) -> list[int]:
    return [_map_one(value, mapping, label) for value in values]


def _remap_payload_references(event: JsonObject, mapping: dict[int, int]) -> JsonObject:
    data = require_record(event.get("data"), f"{event['type']} data")
    if event["type"] == "command/done":
        if data.get("sourceEventSeq") is None:
            return data
        return {
            **data,
            "sourceEventSeq": _map_one(
                data["sourceEventSeq"],
                mapping,
                f"command/done {event.get('seq')} sourceEventSeq",
            ),
        }
    if event["type"] in ("compaction/prune", "compaction/summary"):
        range_value = require_record(data.get("shadowedRange"), "shadowedRange")
        return {
            **data,
            "shadowedRange": {
                "start": _map_one(
                    range_value["start"],
                    mapping,
                    f"{event['type']} {event.get('seq')} shadowedRange start",
                ),
                "end": _map_one(
                    range_value["end"],
                    mapping,
                    f"{event['type']} {event.get('seq')} shadowedRange end",
                ),
            },
            "shadowedSeqs": _map_list(
                data["shadowedSeqs"],
                mapping,
                f"{event['type']} {event.get('seq')} shadowedSeqs",
            ),
        }
    if event["type"] in ("session/title", "session/title-llm-request"):
        return {
            **data,
            "messageSeqs": _map_list(
                data["messageSeqs"],
                mapping,
                f"{event['type']} {event.get('seq')} messageSeqs",
            ),
        }
    return data


def _remap_references(
    source: JsonObject, target_seq: int, mapping: dict[int, int]
) -> JsonObject:
    event = {
        k: v for k, v in source.items() if k not in ("sourceEventSeqs", "surfaceOp")
    }
    sources: JsonObject = {}
    if source.get("sourceEventSeqs") is not None:
        sources = {
            "sourceEventSeqs": _map_list(
                source["sourceEventSeqs"],
                mapping,
                f"{source['type']} {source.get('seq')} sources",
            )
        }
    operation: Any = source.get("surfaceOp")
    if operation is not None and operation != "append":
        replacement = require_record(operation, "surfaceOp")
        operation = {
            "op": "replace",
            "start": _map_one(
                replacement["start"],
                mapping,
                f"{source['type']} {source.get('seq')} surface start",
            ),
            "end": _map_one(
                replacement["end"],
                mapping,
                f"{source['type']} {source.get('seq')} surface end",
            ),
        }
    return {
        **event,
        "seq": target_seq,
        "data": _remap_payload_references(source, mapping),
        **sources,
        **({} if operation is None else {"surfaceOp": operation}),
    }


def _ensure_target_cut(
    state: _V1ToV2State, origin: int, time: int, event_type: str, emit: _Emit
) -> None:
    if (
        not state.source_header["isSeeded"]
        or state.target_cut is not None
        or origin < state.source_cut
    ):
        return
    state.target_cut = state.target_seq
    if origin == state.source_cut and event_type == "session/end-seed":
        return
    emit.emit_event(
        {
            "type": "session/end-seed",
            "seq": state.target_seq,
            "time": time,
            "data": {"inherited": True},
        }
    )
    state.target_seq += 1


def _emit_source(state: _V1ToV2State, event: JsonObject, emit: _Emit) -> None:
    source = event
    if (
        state.source_header["isSeeded"]
        and event["seq"] == state.source_cut
        and event["type"] == "session/end-seed"
    ):
        source = {**event, "data": {"inherited": True}}
    _ensure_target_cut(state, event["seq"], event["time"], source["type"], emit)
    state.mapping[event["seq"]] = state.target_seq
    emit.emit_event(_remap_references(source, state.target_seq, state.mapping))
    state.target_seq += 1


def _emit_generated(
    state: _V1ToV2State, origin: int, event: JsonObject, emit: _Emit
) -> None:
    _ensure_target_cut(state, origin, event["time"], event["type"], emit)
    emit.emit_event(_remap_references(event, state.target_seq, state.mapping))
    state.target_seq += 1


def _message_event(source: JsonObject, group: _AttemptGroup) -> JsonObject:
    data = require_record(source.get("data"), "assistant/message data")
    event = {k: v for k, v in source.items() if k != "sourceEventSeqs"}
    return {**event, "data": {**data, "stream": _stream_of(group)}}


def _attempt_event(group: _AttemptGroup) -> JsonObject:
    return {
        "type": "assistant/attempt",
        "seq": group.last_chunk_seq,
        "time": group.last_chunk_time,
        "data": {"turn": group.turn, "step": group.step, "stream": _stream_of(group)},
    }


def _flush_buffered(state: _V1ToV2State, pending: dict[str, Any], emit: _Emit) -> None:
    for event in pending["after_last_chunk"]:
        _emit_source(state, event, emit)
    pending["after_last_chunk"] = []


def _finish_attempt(state: _V1ToV2State, emit: _Emit) -> None:
    pending = state.pending
    if pending is None:
        return
    group = pending["group"]
    _emit_generated(state, group.last_chunk_seq, _attempt_event(group), emit)  # type: ignore[arg-type]
    _flush_buffered(state, pending, emit)
    state.pending = None


def _assert_attempt_cut(state: _V1ToV2State, group: _AttemptGroup, member: int) -> None:
    first = group.spans[0]["first_seq"] if group.spans else member
    if (first < state.source_cut) != (member < state.source_cut):
        raise _v1_v2_refusal(
            f"inherited Session cut {state.source_cut} splits one Assistant attempt"
        )


def _assert_attempt_range(state: _V1ToV2State, first: int, last: int) -> None:
    if (first < state.source_cut) != (last < state.source_cut):
        raise _v1_v2_refusal(
            f"inherited Session cut {state.source_cut} splits one Assistant attempt"
        )


def _legacy_interrupted_turn(
    state: _LegacyTurnState, event: JsonObject
) -> JsonObject | None:
    if (
        event.get("type") != "turn/start"
        or state.open_turn is None
        or state.open_step is not None
        or require_record(event.get("data"), "turn/start data").get("turn")
        != state.open_turn + 1
        or state.previous is None
        or state.previous.get("type") != "agent/inbox/spliced"
    ):
        return None
    splice = require_record(state.previous.get("data"), "splice data")
    if (
        splice.get("target") != "next-turn"
        or not isinstance(splice.get("inserted"), list)
        or not splice["inserted"]
    ):
        return None
    return {
        "type": "turn/end",
        "seq": event["seq"],
        "time": event["time"],
        "data": {"turn": state.open_turn, "reason": {"kind": "interrupted"}},
    }


def _observe_legacy_turn(state: _LegacyTurnState, event: JsonObject) -> None:
    data = event.get("data")
    data_record = data if is_json_object(data) else {}
    if event.get("type") == "turn/start":
        state.open_turn = require_count(data_record.get("turn"), "turn")
        state.open_step = None
    elif event.get("type") == "turn/end":
        state.open_turn = None
        state.open_step = None
    elif event.get("type") == "step/start":
        state.open_step = require_count(data_record.get("step"), "step")
    elif event.get("type") == "step/end":
        state.open_step = None
    state.previous = event


def _split_legacy_goal_change(
    event: JsonObject,
) -> tuple[JsonObject, JsonObject] | None:
    if event.get("type") != "user/message":
        return None
    data = require_record(event.get("data"), "user/message data")
    source = require_record(data.get("source"), "user/message source")
    if source.get("kind") != "goal" or "change" not in source:
        return None
    change = {
        "type": "goal/change",
        "seq": event["seq"],
        "time": event["time"],
        "data": source["change"],
    }
    message = {
        **event,
        "data": {**data, "source": {"kind": "plugin", "plugin": "goal"}},
    }
    return change, message


def _closes_attempt(event: JsonObject) -> bool:
    return event.get("type") in (
        "turn/end",
        "step/end",
        "llm/retry",
        "llm/retry-started",
    )


def _assert_chunk_envelope(event: JsonObject) -> None:
    unexpected = next((key for key in event if key not in _CHUNK_EVENT_KEYS), None)
    if unexpected is not None:
        raise _v1_v2_refusal(
            f"assistant/chunk {event.get('seq')} has unexpected member {unexpected}"
        )
    missing = next((key for key in _CHUNK_EVENT_REQUIRED if key not in event), None)
    if missing is not None:
        raise _v1_v2_refusal(
            f"assistant/chunk {event.get('seq')} lacks required member {missing}"
        )
    if event.get("ignorable") is not None and event["ignorable"] is not True:
        raise _v1_v2_refusal(
            f"assistant/chunk {event.get('seq')} ignorable must be true when present"
        )


def _assert_source_delivery_marker_v1(state: _V1ToV2State, event: JsonObject) -> None:
    if event.get("type") != "session-log-deepseek/delivery-accepted":
        return
    data = require_record(event.get("data"), "delivery-accepted data")
    inherited = (
        state.source_header.get("parentSession") is not None
        and event["seq"] < state.source_cut
    )
    if (
        data.get("sessionFormatVersion") == 1
        and not inherited
        and data.get("sessionId") != state.source_header.get("id")
    ):
        raise _v1_v2_refusal(
            "current-generation delivery marker names the wrong Session"
        )


def _transform_chunk(state: _V1ToV2State, event: JsonObject, emit: _Emit) -> None:
    data = require_record(event.get("data"), "assistant/chunk data")
    turn = require_count(data.get("turn"), "turn")
    step = require_count(data.get("step"), "step")
    chunk = require_record(data.get("chunk"), "assistant/chunk chunk")
    pending = state.pending
    if pending is not None and (
        pending["group"].terminal
        or pending["group"].turn != turn
        or pending["group"].step != step
    ):
        _finish_attempt(state, emit)
        pending = None
    elif pending is not None:
        _flush_buffered(state, pending, emit)
    if pending is None:
        pending = {"group": _AttemptGroup(turn, step), "after_last_chunk": []}
        state.pending = pending
    group = pending["group"]
    _assert_attempt_cut(state, group, event["seq"])
    if group.accumulator is None:
        group.accumulator = AssistantStreamAccumulator()
    group.accumulator.push(event["time"], chunk)
    _record_chunk_span(group, event["seq"], 1, event["time"])
    if chunk.get("type") == "finish":
        group.terminal = True


def _transform_message(state: _V1ToV2State, event: JsonObject, emit: _Emit) -> None:
    data = require_record(event.get("data"), "assistant/message data")
    turn = require_count(data.get("turn"), "turn")
    step = require_count(data.get("step"), "step")
    sources = event.get("sourceEventSeqs")
    pending = state.pending
    if pending is not None and (
        pending["group"].turn != turn or pending["group"].step != step
    ):
        _finish_attempt(state, emit)
        _emit_source(state, _message_event(event, _AttemptGroup(turn, step)), emit)
        return
    if sources is None:
        if pending is not None:
            raise _v1_v2_refusal(
                f"assistant/message {event.get('seq')} does not cite its complete "
                "v1 chunk attempt"
            )
        _emit_source(state, _message_event(event, _AttemptGroup(turn, step)), emit)
        return
    if isinstance(sources, list) and not sources:
        _finish_attempt(state, emit)
        _emit_source(state, _message_event(event, _AttemptGroup(turn, step)), emit)
        return
    if pending is None or not _matches_chunk_sources(pending["group"], sources):
        raise _v1_v2_refusal(
            f"assistant/message {event.get('seq')} chunk references are not one "
            "complete ordered attempt"
        )
    _assert_attempt_cut(state, pending["group"], event["seq"])
    pending["group"].terminal = True
    _flush_buffered(state, pending, emit)
    _emit_source(state, _message_event(event, pending["group"]), emit)
    state.pending = None


def _transform_released_event(
    state: _V1ToV2State, event: JsonObject, emit: _Emit
) -> None:
    if event.get("type") == "assistant/chunk":
        _assert_chunk_envelope(event)
    if event.get("type") not in RELEASED_V0_EVENT_DISPOSITIONS:
        raise _v1_v2_refusal(
            f"format v1 contains unknown event type "
            f"{json.dumps(event.get('type'))} at seq {event.get('seq')}"
        )
    interrupted = _legacy_interrupted_turn(state.legacy_turns, event)
    if (
        event.get("type") == "turn/start"
        and state.legacy_turns.open_turn is not None
        and interrupted is None
    ):
        turn = require_record(event.get("data"), "turn/start data").get("turn")
        raise _v1_v2_refusal(
            f"turn/start {json.dumps(turn)} does not close the prior turn"
        )
    _assert_source_delivery_marker_v1(state, event)
    _observe_legacy_turn(state.legacy_turns, event)
    state.last_time = event["time"]
    if interrupted is not None:
        _finish_attempt(state, emit)
        _emit_generated(state, event["seq"], interrupted, emit)
    legacy_goal = _split_legacy_goal_change(event)
    if legacy_goal is not None:
        change, message = legacy_goal
        _emit_generated(state, event["seq"], change, emit)
        _emit_source(state, message, emit)
        return
    if event.get("type") == "assistant/chunk":
        _transform_chunk(state, event, emit)
        return
    if event.get("type") == "assistant/message":
        _transform_message(state, event, emit)
        return
    if _closes_attempt(event):
        _finish_attempt(state, emit)
        _emit_source(state, event, emit)
        return
    if state.pending is not None:
        state.pending["after_last_chunk"].append(event)
        return
    _emit_source(state, event, emit)


def _transform_released_run(
    state: _V1ToV2State, run: SessionFormatEventRun, emit: _Emit
) -> None:
    if not isinstance(run, PackedAssistantChunkRun):
        for event in run.expand():
            _transform_released_event(state, event, emit)
        return
    state.legacy_turns.previous = None
    state.last_time = run.last_time
    pending = state.pending
    if pending is not None and (
        pending["group"].terminal
        or pending["group"].turn != run.turn
        or pending["group"].step != run.step
    ):
        _finish_attempt(state, emit)
        pending = None
    elif pending is not None:
        _flush_buffered(state, pending, emit)
    if pending is None:
        pending = {"group": _AttemptGroup(run.turn, run.step), "after_last_chunk": []}
        state.pending = pending
    group = pending["group"]
    _assert_attempt_range(state, run.first_seq, run.last_seq)
    _flush_accumulator(group)
    _append_stream_record(group, dict(run.stream), run.last_time)
    _record_chunk_span(group, run.first_seq, run.event_count, run.last_time)


def _finish_migration_v1_to_v2(state: _V1ToV2State, emit: _Emit) -> int:
    _finish_attempt(state, emit)
    if state.source_header["isSeeded"] and state.target_cut is None:
        state.target_cut = state.target_seq
        emit.emit_event(
            {
                "type": "session/end-seed",
                "seq": state.target_seq,
                "time": state.last_time,
                "data": {"inherited": True},
            }
        )
        state.target_seq += 1
    return state.target_cut  # type: ignore[return-value]


class _V1ToV2Stage(_Stage):
    """Embed released-v1 assistant chunks into v2 attempt events."""

    def __init__(self, input_: SessionFormatMigrationStageInput) -> None:
        assert_v1_header(input_.source_header)
        self._state = _V1ToV2State(
            input_.source_header,
            require_count(
                input_.source_inherited_event_count,
                "format v1 inherited event count",
            ),
        )

    def transform_event(self, event: JsonObject, emit: _Emit) -> None:
        _transform_released_event(self._state, event, emit)

    def transform_run(self, run: SessionFormatEventRun, emit: _Emit) -> None:
        _transform_released_run(self._state, run, emit)

    def finish(self, emit: _Emit) -> int:
        return _finish_migration_v1_to_v2(self._state, emit)


class _DecodedV1ToV2Stage(_V1ToV2Stage):
    """Decoded v1 stage: validates payloads before transforming."""

    def transform_event(self, event: JsonObject, emit: _Emit) -> None:
        if (
            event.get("type") != "assistant/chunk"
            and event.get("type") in RELEASED_V0_EVENT_DISPOSITIONS
        ):
            assert_released_event_payload(event, 1)
        super().transform_event(event, emit)


def _v1_to_v2_migrate_header(header: JsonObject) -> JsonObject:
    assert_v1_header(header)
    return {**header, "version": 2}


def _v1_to_v2_create_stage(
    input_: SessionFormatMigrationStageInput,
) -> _Stage:
    if input_.source_kind == "decoded":
        return _DecodedV1ToV2Stage(input_)
    return _V1ToV2Stage(input_)


SESSION_FORMAT_V1_TO_V2 = SessionFormatMigration(
    name="session-format-v1-to-v2",
    from_version=1,
    to_version=2,
    migrate_header=_v1_to_v2_migrate_header,
    create_stage=_v1_to_v2_create_stage,
    validate_target_header=assert_v2_header,
)

# ---------------------------------------------------------------------------
# Migration edge v2 -> v3
# ---------------------------------------------------------------------------

_SOURCE_KINDS = frozenset(
    [
        "user",
        "plugin",
        "model",
        "tool",
        "agent-instructions",
        "session-reference",
        "team-message",
        "goal",
        "skill-invocation",
        "skill-catalog",
        "coordinator",
        "subagent-report",
        "subagent-settled",
        "webhook",
        "agent-message",
    ]
)


def _is_repair_identity(message_id: Any, call_id: Any) -> bool:
    if not isinstance(call_id, str):
        return False
    prefix = "interrupted-tool-result-" + call_id + "-"
    if not isinstance(message_id, str) or not message_id.startswith(prefix):
        return False
    suffix = message_id[len(prefix) :]
    if not suffix or (len(suffix) > 1 and suffix.startswith("0")):
        return False
    if not suffix.isdigit():
        return False
    return 0 <= int(suffix) <= MAX_SAFE_INTEGER


def _assert_content_kind(kind: Any, label: str) -> None:
    if not isinstance(kind, str) or kind not in CORE_CONTENT_TYPES:
        raise SessionFormatUnsupportedMigrationError(
            f"{label}: cannot safely transform unclassified message content kind "
            f"{json.dumps(kind)}"
        )


def _assert_content_block(value: Any, label: str) -> None:
    block = require_record(value, label)
    _assert_content_kind(block.get("type"), label)
    if block["type"] == "tool-result":
        if not isinstance(block.get("content"), list):
            raise SessionFormatError(
                f'{label}.content: invalid message content kind "tool-result": '
                "content must be an array"
            )
        for child in block["content"]:
            _assert_content_block(child, f"{label}.content")
    if block["type"] == "file":
        exact_keys(block, ["type", "attachment"], [], label + ' kind "file"')
        attachment = require_record(
            block.get("attachment"), label + ' kind "file" attachment'
        )
        exact_keys(
            attachment,
            ["attachmentId", "name", "bytes"],
            [],
            label + ' kind "file" attachment',
        )
        if (
            not isinstance(attachment.get("attachmentId"), str)
            or not attachment["attachmentId"]
        ):
            raise SessionFormatError(
                label + ' kind "file": file attachment requires attachmentId and name'
            )
        if not isinstance(attachment.get("name"), str):
            raise SessionFormatError(
                label + ' kind "file": file attachment requires attachmentId and name'
            )
        require_count(attachment.get("bytes"), label + ' kind "file" attachment bytes')


def _assert_content_kinds(content: Any, label: str) -> None:
    if not isinstance(content, list):
        raise SessionFormatError(f"{label}: content must be an array")
    for index, value in enumerate(content):
        _assert_content_block(value, f"{label}[{index}]")


def _assert_owned_content(event: JsonObject, data: JsonObject) -> None:
    label = f"format v2 {event['type']} at seq {event['seq']} data"
    if event["type"] in ("user/message", "tool/code-dispatch"):
        _assert_content_kinds(data.get("content"), label + ".content")
    elif event["type"] in ("assistant/message", "tool/result", "team/message/queued"):
        _assert_content_kinds(
            require_record(data.get("message"), label + ".message").get("content"),
            label + ".message.content",
        )
    elif event["type"] in ("agent/inbox/spliced", "session/title-llm-request"):
        field_name = (
            "inserted" if event["type"] == "agent/inbox/spliced" else "messages"
        )
        messages = data.get(field_name)
        if not isinstance(messages, list):
            raise SessionFormatError(f"{label}.{field_name}: content must be an array")
        for index, value in enumerate(messages):
            path = f"{label}.{field_name}[{index}]"
            _assert_content_kinds(
                require_record(value, path).get("content"), path + ".content"
            )
    elif event["type"] == "compaction/summary":
        _assert_content_kinds(data.get("summary"), label + ".summary")
        if data.get("rawOutput") is not None:
            _assert_content_kinds(data.get("rawOutput"), label + ".rawOutput")
    if event["type"] in ("assistant/message", "assistant/attempt"):
        stream = data.get("stream")
        if not isinstance(stream, list):
            raise SessionFormatError(f"{label}.stream: content must be an array")
        for index, value in enumerate(stream):
            path = f"{label}.stream[{index}]"
            entry = require_record(value, path)
            if entry.get("type") != "chunk":
                continue
            chunk = require_record(entry.get("chunk"), path + ".chunk")
            if chunk.get("type") == "block-end":
                _assert_content_block(chunk.get("block"), path + ".chunk.block")
            if chunk.get("type") == "block-start":
                _assert_content_kind(chunk.get("blockType"), path + ".chunk.blockType")


def _assert_message_source_v2(message: JsonObject) -> None:
    source = require_record(message.get("source"), "message source")
    if not isinstance(source.get("kind"), str) or source["kind"] not in _SOURCE_KINDS:
        raise SessionFormatUnsupportedMigrationError(
            "cannot safely transform unclassified message source"
        )
    if source["kind"] == "agent-message":
        exact_keys(
            source, ["kind", "form", "senderSessionId"], [], "agent-message source"
        )
        if (
            source.get("form") != "relay"
            or not isinstance(source.get("senderSessionId"), str)
            or not source["senderSessionId"]
        ):
            raise SessionFormatError(
                "agent-message source requires relay form and senderSessionId"
            )


def _assert_event_v2(event: JsonObject) -> None:
    disposition = RELEASED_V2_EVENT_DISPOSITIONS.get(str(event.get("type")))
    feedback = event.get("type") in ("feedback/message-put", "feedback/message-delete")
    if disposition is None and not feedback:
        raise SessionFormatUnsupportedMigrationError(
            "format v2 to v3 cannot safely transform unclassified event "
            f"{event.get('type')}"
        )
    surface = event.get("type") in SURFACE_TYPES
    exact_keys(
        event,
        ["type", "seq", "time", "data"],
        ["ignorable", "sourceEventSeqs", "surfaceOp"] if surface else ["ignorable"],
        str(event.get("type")),
    )
    require_count(event.get("seq"), "event seq")
    require_safe_integer(event.get("time"), "event time")
    if event.get("ignorable") is not None and event["ignorable"] is not True:
        raise SessionFormatError("ignorable must be true")
    if surface:
        assert_released_surface_metadata(
            event, event["seq"], str(event["type"]), "forbid-assistant"
        )
        if event.get("surfaceOp") is None:
            raise SessionFormatError(f"{event['type']} requires surfaceOp")
    data = require_record(event.get("data"), f"{event['type']} data")
    if feedback:
        if event["type"] == "feedback/message-put":
            exact_keys(data, ["sessionId", "item"], [], str(event["type"]))
        else:
            exact_keys(data, ["sessionId", "messageId"], [], str(event["type"]))
        if not isinstance(data.get("sessionId"), str):
            raise SessionFormatError("feedback sessionId must be a string")
        if event["type"] == "feedback/message-delete":
            if not isinstance(data.get("messageId"), str):
                raise SessionFormatError("feedback messageId must be a string")
            return
        item = require_record(data.get("item"), "feedback item")
        exact_keys(
            item,
            ["messageId", "rating", "version", "createdAt", "updatedAt"],
            ["note"],
            "feedback item",
        )
        for key in ("messageId", "version"):
            if not isinstance(item.get(key), str):
                raise SessionFormatError(f"feedback {key} must be a string")
        if item.get("rating") not in ("positive", "negative"):
            raise SessionFormatError("invalid feedback rating")
        if item.get("note") is not None and not isinstance(item["note"], str):
            raise SessionFormatError("feedback note must be a string")
        require_count(item.get("createdAt"), "feedback createdAt")
        require_count(item.get("updatedAt"), "feedback updatedAt")
        return
    assert disposition is not None
    exact_keys(
        data,
        disposition.required,
        disposition.optional,
        f"{event['type']} data",
    )
    _assert_owned_content(event, data)
    if event["type"] in ("assistant/message", "assistant/attempt"):
        for coordinate in ("turn", "step"):
            if require_count(data.get(coordinate), coordinate) == 0:
                raise SessionFormatError(f"{coordinate} must be positive")
    if (
        event["type"] == "session/end-seed"
        and data.get("inherited") is not None
        and data["inherited"] is not True
    ):
        raise SessionFormatError("session/end-seed inherited must be true")
    if event["type"] == "user/message":
        _assert_message_source_v2(data)
    if event["type"] in ("assistant/message", "tool/result"):
        _assert_message_source_v2(require_record(data.get("message"), "message"))
    if (
        event["type"] == "tool/result"
        and is_json_object(data.get("error"))
        and data["error"].get("code") == "TOOL_NOT_STARTED"
    ):
        message = require_record(data.get("message"), "tool result message")
        source = require_record(message.get("source"), "tool result source")
        if not _is_repair_identity(message.get("id"), source.get("callId")):
            raise SessionFormatError(
                "TOOL_NOT_STARTED repair requires its canonical historical message id"
            )
    if event["type"] in ("agent/inbox/spliced", "session/title-llm-request"):
        field_name = (
            "inserted" if event["type"] == "agent/inbox/spliced" else "messages"
        )
        messages = data.get(field_name)
        # pragma: no cover - defensive only: _assert_owned_content above
        # raises for a non-list field or a non-object member, so both False
        # edges below are unreachable through this function.
        if isinstance(messages, list):  # pragma: no cover
            for message in messages:
                if is_json_object(message):  # pragma: no cover
                    _assert_message_source_v2(message)


def _remap_event_v2_to_v3(
    event: JsonObject, seq: int, mapping: list[int]
) -> JsonObject:
    def one(value: Any) -> int:
        source = require_count(value, "source event reference")
        if source >= event["seq"] or source >= len(mapping):
            raise SessionFormatError("reference must name an earlier source event")
        return mapping[source]

    def values(value: Any) -> list[int]:
        if not isinstance(value, list):
            raise SessionFormatError("sequence references must be an array")
        return [one(item) for item in value]

    def range_value(value: Any) -> JsonObject:
        source = require_record(value, "sequence range")
        return {**source, "start": one(source["start"]), "end": one(source["end"])}

    data = require_record(event.get("data"), str(event.get("type")))
    if event["type"] == "command/done":
        if data.get("sourceEventSeq") is not None:
            data = {**data, "sourceEventSeq": one(data["sourceEventSeq"])}
    elif event["type"] in ("compaction/summary", "compaction/prune"):
        data = {
            **data,
            "shadowedRange": range_value(data.get("shadowedRange")),
            "shadowedSeqs": values(data.get("shadowedSeqs")),
        }
    elif event["type"] in ("session/title", "session/title-llm-request"):
        data = {**data, "messageSeqs": values(data.get("messageSeqs"))}
    event_out: JsonObject = {
        **event,
        "seq": seq,
        "data": data,
    }
    if event.get("sourceEventSeqs") is not None:
        event_out["sourceEventSeqs"] = values(event["sourceEventSeqs"])
    if event.get("surfaceOp") is not None and event["surfaceOp"] != "append":
        event_out["surfaceOp"] = range_value(event["surfaceOp"])
    return event_out


def _canonicalize_transformed_event(event: JsonObject) -> JsonObject:
    target = event
    operation = event.get("surfaceOp")
    if operation is not None and operation != "append":
        replace = require_record(
            operation, f"format v2 {event['type']} at seq {event.get('seq')} surfaceOp"
        )
        if (
            len(replace) != 3
            or replace.get("op") != "replace"
            or "start" not in replace
            or "end" not in replace
        ):
            raise SessionFormatError(
                f"format v2 {event['type']} at seq {event.get('seq')} requires exact "
                "replace fields op/start/end"
            )
        target = {
            **event,
            "surfaceOp": {
                "op": "replace",
                "startSeq": require_count(
                    replace["start"],
                    f"format v2 {event['type']} at seq {event.get('seq')} replace start",
                ),
                "endSeq": require_count(
                    replace["end"],
                    f"format v2 {event['type']} at seq {event.get('seq')} replace end",
                ),
            },
        }
    if event["type"] == "request/header":
        data = require_record(
            event.get("data"),
            f"format v2 request/header at seq {event.get('seq')} data",
        )
        header = require_record(
            data.get("header"),
            f"format v2 request/header at seq {event.get('seq')} header",
        )
        empty = [
            key
            for key in header
            if (key == "tools" and isinstance(header[key], list) and not header[key])
            or (
                key == "adapterDefaults"
                and is_json_object(header[key])
                and not header[key]
            )
        ]
        if empty:
            canonical = {
                key: value for key, value in header.items() if key not in empty
            }
            target = {**target, "data": {**data, "header": canonical}}
    assert_v3_event(target)
    return target


def _rename_ptc_event(event: JsonObject) -> JsonObject:
    if event["type"] == "agent-preset/selected":
        data = require_record(event.get("data"), "agent-preset/selected data")
        if data.get("agentPreset") == "code":
            return {**event, "data": {**data, "agentPreset": "ptc"}}
        return event
    if event["type"] == "tool/code-dispatch-start":
        return {**event, "type": "tool/ptc-dispatch-start"}
    if event["type"] == "tool/code-dispatch":
        return {**event, "type": "tool/ptc-dispatch"}

    def rename_source(message: JsonObject) -> JsonObject:
        source = message.get("source")
        if (
            not is_json_object(source)
            or source.get("kind") != "plugin"
            or source.get("plugin") != "tools-code-mode"
        ):
            return message
        return {**message, "source": {**source, "plugin": "tools-ptc"}}

    if event["type"] == "user/message":
        data = require_record(event.get("data"), "user/message data")
        renamed = rename_source(data)
        return event if renamed is data else {**event, "data": renamed}
    if event["type"] in ("agent/inbox/spliced", "session/title-llm-request"):
        data = require_record(event.get("data"), f"{event['type']} data")
        key = "inserted" if event["type"] == "agent/inbox/spliced" else "messages"
        messages = data.get(key)
        if not isinstance(messages, list):
            return event
        renamed_messages = [
            rename_source(message) if is_json_object(message) else message
            for message in messages
        ]
        if all(
            renamed_messages[i] is messages[i] for i in range(len(renamed_messages))
        ):
            return event
        return {**event, "data": {**data, key: renamed_messages}}
    return event


class _V2ToV3Stage(_Stage):
    """Promote system prompts, remap references, canonicalize v3 envelopes."""

    def __init__(self, input_: SessionFormatMigrationStageInput) -> None:
        assert_v2_header(input_.source_header)
        self._input = input_
        self._mapping: list[int] = []
        self._original_ids: set[Any] = set()
        self._generated_ids: set[str] = set()
        self._target_seq = 0
        self._source_cut: int | None = None if input_.source_header["isSeeded"] else 0
        self._target_cut: int | None = None if input_.source_header["isSeeded"] else 0
        self._last_foreign_delivery_seq: int | None = None
        self._step: dict[str, int] | None = None
        self._head: int | None = None
        self._prompt = ""
        self.header_inherited_event_count = (
            0 if not input_.source_header["isSeeded"] else None
        )

    def _observe_message_ids(self, event: JsonObject) -> None:
        data = require_record(event.get("data"), str(event.get("type")))
        if event["type"] == "user/message":
            messages: list[Any] = [data]
        elif event["type"] in ("assistant/message", "tool/result"):
            messages = [require_record(data.get("message"), "message")]
        elif event["type"] == "agent/inbox/spliced":
            messages = data.get("inserted", [])
        elif event["type"] == "session/title-llm-request":
            messages = data.get("messages", [])
        else:
            messages = []
        for message in messages:
            if not is_json_object(message):
                continue
            message_id = message.get("id")
            if message_id in self._generated_ids:
                raise SessionFormatUnsupportedMigrationError(
                    "source message id collides with a generated system message id"
                )
            self._original_ids.add(message_id)

    def _emit_system(self, prompt: str, anchor: JsonObject, emit: _Emit) -> None:
        if self._step is None:
            raise SessionFormatUnsupportedMigrationError(
                "format v2 changed request prompt outside an open step cannot "
                "retain source chronology"
            )
        identity = json.dumps(
            [
                "session-format-v2-to-v3",
                self._input.source_header["id"],
                anchor["seq"],
                anchor["type"],
            ],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        message_id = (
            "v2-to-v3-system-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()
        )
        if message_id in self._original_ids or message_id in self._generated_ids:
            raise SessionFormatUnsupportedMigrationError(
                "generated system message id collides with an existing message id"
            )
        self._generated_ids.add(message_id)
        seq = self._target_seq
        self._target_seq += 1
        step = self._step
        system_event: JsonObject = {
            "type": "system/message",
            "seq": seq,
            "time": anchor["time"],
            "data": {
                **step,
                "message": {
                    "id": message_id,
                    "role": "system",
                    "source": {
                        "kind": "plugin",
                        "plugin": "@deepseek-ai/dsh-system-prompt",
                    },
                    "content": (
                        [] if prompt == "" else [{"type": "text", "text": prompt}]
                    ),
                },
            },
            **(
                {"surfaceOp": "append"}
                if self._head is None
                else {
                    "surfaceOp": {
                        "op": "replace",
                        "start": self._head,
                        "end": self._head,
                    },
                    "sourceEventSeqs": [self._head],
                }
            ),
        }
        emit.emit_event(_canonicalize_transformed_event(system_event))
        self._head = seq
        self._prompt = prompt

    def transform_event(self, event: JsonObject, emit: _Emit) -> None:
        if event["seq"] != len(self._mapping):
            raise SessionFormatError("format v2 source events must be dense")
        _assert_event_v2(event)
        self._observe_message_ids(event)
        source = event
        data = require_record(event.get("data"), str(event.get("type")))
        if event["type"] == "request/header":
            header = require_record(data.get("header"), "request header")
            system = header.get("system")
            prompt = system if isinstance(system, str) else ""
            if prompt != self._prompt:
                self._emit_system(prompt, event, emit)
            source = {
                **event,
                "data": {
                    **data,
                    "header": {k: v for k, v in header.items() if k != "system"},
                },
            }
        if event["type"] in SURFACE_TYPES and self._head is None:
            raise SessionFormatUnsupportedMigrationError(
                "format v2 surface before first step cannot acquire a system head "
                "without changing chronology"
            )
        if event["type"] == "session/end-seed" and data.get("inherited") is True:
            if not self._input.source_header["isSeeded"]:
                raise SessionFormatError(
                    "format v2 unseeded Session contains an inherited end-seed marker"
                )
            self._source_cut = event["seq"]
            self._target_cut = self._target_seq
        if event["type"] == "session-log-deepseek/delivery-accepted":
            if data.get("sessionFormatVersion") == 3:
                raise SessionFormatError(
                    "format v2 delivery marker claims target format v3"
                )
            if (
                data.get("sessionFormatVersion") == 2
                and data.get("sessionId") != self._input.source_header["id"]
            ):
                self._last_foreign_delivery_seq = event["seq"]
        target = _remap_event_v2_to_v3(source, self._target_seq, self._mapping)
        self._mapping.append(self._target_seq)
        self._target_seq += 1
        emit.emit_event(_canonicalize_transformed_event(_rename_ptc_event(target)))
        if event["type"] == "step/start":
            self._step = {
                "turn": data["turn"],
                "step": data["step"],
            }
            if self._head is None:
                self._emit_system("", event, emit)
        elif event["type"] in ("step/end", "turn/end"):
            self._step = None

    def finish(self, emit: _Emit) -> int:
        cut = require_count(self._source_cut, "format v2 inherited end-seed marker")
        if (
            self._input.source_inherited_event_count is not None
            and self._input.source_inherited_event_count != cut
        ):
            raise SessionFormatError(
                "format v2 inherited end-seed marker disagrees with its source cut"
            )
        if self._last_foreign_delivery_seq is not None and (
            self._input.source_header.get("parentSession") is None
            or self._last_foreign_delivery_seq >= cut
        ):
            raise SessionFormatError(
                "current-generation delivery marker names the wrong Session"
            )
        return require_count(self._target_cut, "format v3 inherited event count")


def _v2_to_v3_migrate_header(header: JsonObject) -> JsonObject:
    assert_v2_header(header)
    migrated = {**header, "version": 3}
    if header.get("agentPreset") == "code":
        migrated["agentPreset"] = "ptc"
    return migrated


SESSION_FORMAT_V2_TO_V3 = SessionFormatMigration(
    name="session-format-v2-to-v3",
    from_version=2,
    to_version=3,
    migrate_header=_v2_to_v3_migrate_header,
    create_stage=lambda input_: _V2ToV3Stage(input_),
    validate_target_header=assert_v3_header,
)

# ---------------------------------------------------------------------------
# Migration edge v3 -> v4
# ---------------------------------------------------------------------------

# Released v3 plugin names whose current producer kind is not the plugin
# string. These are wire-format plugin identities from the format
# specification, kept verbatim so historical sources rewrite exactly; they
# are data values, not references to any project.
_RENAMED_PRODUCERS: dict[str, str] = {
    "compact": "compact-checkpoint",
    "tools-code-mode": "ptc-mode",
    "tools-ptc": "ptc-mode",
    "dsh-compaction-basic": "compact-basic",
    "@deepseek-ai/dsh-system-prompt": "runtime-context",
}

# First-party v3 plugin identities that intentionally keep their current kind.
_RELEASED_SAME_NAME_PRODUCERS: frozenset[str] = frozenset(
    [
        "agent-instructions",
        "session-reference",
        "team-message",
        "goal",
        "skill-invocation",
        "skill-catalog",
        "coordinator",
        "subagent-report",
        "subagent-settled",
        "webhook",
        "agent-message",
        "model-selection",
        "plan-mode",
        "time-context",
        "tmux-context",
        "user-approval",
        "repeat-tool-reminder",
        "tool-cordis",
        "cordis-host-runner",
        "tool-goal",
        "tool-jobs",
        "hooks-codex",
        "hooks-claude-code",
        "schedule",
        "dsh-session-title-llm",
    ]
)


def _producer_kind(plugin: str, role: Any) -> str:
    if plugin == "@deepseek-ai/dsh-system-prompt" and role == "system":
        return "system-prompt"
    renamed = _RENAMED_PRODUCERS.get(plugin)
    if renamed is not None:
        return renamed
    if plugin in _RELEASED_SAME_NAME_PRODUCERS:
        return plugin
    return f"plugin:{plugin}"


def _rewrite_plugin_source(source: JsonObject, seq: int, role: Any) -> JsonObject:
    plugin = source.get("plugin")
    if not isinstance(plugin, str):
        raise SessionFormatError(
            f"plugin source at seq {seq} is not canonical: plugin requires a string"
        )
    kind = _producer_kind(plugin, role)
    if len(source) == 2:
        return {"kind": kind}
    return {
        key: (kind if key == "kind" else value)
        for key, value in source.items()
        if key != "plugin"
    }


def rewrite_v3_message_source(source: JsonObject, seq: int, role: Any) -> JsonObject:
    """Convert released plugin wrappers, retaining direct source kinds."""
    kind = source.get("kind")
    if not isinstance(kind, str) or not kind:
        raise SessionFormatError(
            f"message source at seq {seq} requires a nonempty kind"
        )
    if kind == "plugin":
        return _rewrite_plugin_source(source, seq, role)
    return source


def namespace_v3_opaque_event(event: JsonObject) -> JsonObject:
    """Keep unknown ignorable v3 events opaque after header promotion."""
    if event.get("ignorable") is True and event.get("type") not in (
        RELEASED_V3_EVENT_TYPES
    ):
        return {**event, "type": f"plugin:{event['type']}", "ignorable": True}
    return event


def _remap_v3_references(event: JsonObject, seq: int, mapping: list[int]) -> JsonObject:
    if seq == event["seq"]:
        return event

    def reference(value: Any) -> int:
        source = require_count(value, "V3 source event reference")
        target = mapping[source] if 0 <= source < len(mapping) else None
        if source >= event["seq"] or target is None:
            raise SessionFormatError("V3 reference must name an earlier source event")
        return target

    def references(value: Any) -> list[int]:
        if not isinstance(value, list):
            raise SessionFormatError("V3 event references must be an array")
        return [reference(item) for item in value]

    data = event.get("data")
    if event["type"] == "command/done":
        source = require_record(data, "command/done data")
        if source.get("sourceEventSeq") is not None:
            data = {
                **source,
                "sourceEventSeq": reference(source["sourceEventSeq"]),
            }
    elif event["type"] in ("compaction/summary", "compaction/prune"):
        source = require_record(data, f"{event['type']} data")
        range_value = require_record(source.get("shadowedRange"), "shadowedRange")
        data = {
            **source,
            "shadowedRange": {
                **range_value,
                "start": reference(range_value["start"]),
                "end": reference(range_value["end"]),
            },
            "shadowedSeqs": references(source.get("shadowedSeqs")),
        }
    elif event["type"] in ("session/title", "session/title-llm-request"):
        source = require_record(data, f"{event['type']} data")
        data = {**source, "messageSeqs": references(source.get("messageSeqs"))}
    elif event["type"] == "image/offload":
        source = require_record(data, "image/offload data")
        targets = source.get("targets")
        if not isinstance(targets, list):
            raise SessionFormatError("V3 image offload targets must be an array")
        remapped_targets = []
        for target_value in targets:
            target = require_record(target_value, "image offload target")
            remapped_targets.append({**target, "seq": reference(target["seq"])})
        data = {**source, "targets": remapped_targets}
    surface = event.get("surfaceOp")
    range_op = (
        None
        if surface is None or surface == "append"
        else require_record(surface, "surfaceOp")
    )
    return {
        **event,
        "seq": seq,
        "data": data,
        **(
            {}
            if event.get("sourceEventSeqs") is None
            else {"sourceEventSeqs": references(event["sourceEventSeqs"])}
        ),
        **(
            {}
            if range_op is None
            else {
                "surfaceOp": {
                    **range_op,
                    "startSeq": reference(range_op["startSeq"]),
                    "endSeq": reference(range_op["endSeq"]),
                }
            }
        ),
    }


_WRAPPER_FIELDS = frozenset(["type", "toolCallId", "content", "isError"])
_MESSAGE_FIELDS = frozenset(["id", "role", "source", "content"])


def _extension_fields(
    value: JsonObject, fields: frozenset[str], owner: str
) -> JsonObject:
    return {
        f"plugin:{owner}:{key}": field
        for key, field in value.items()
        if key not in fields
    }


def _result_content(value: Any, subject: str) -> list[Any]:
    if not isinstance(value, list):
        raise SessionFormatError(f"{subject} tool-result content must be an array")
    if any(
        is_json_object(block) and block.get("type") == "tool-result" for block in value
    ):
        raise SessionFormatUnsupportedMigrationError(
            f"{subject} contains a nested tool-result unsupported by this converter"
        )
    return value


def lift_tool_result(event: JsonObject) -> JsonObject:
    """Lift one v3 wrapper tool/result row into a first-class v4 message."""
    if event.get("type") != "tool/result" or not is_json_object(event.get("data")):
        return event
    data = event["data"]
    message = data.get("message")
    if not is_json_object(message) or message.get("role") != "user":
        return event
    source = message.get("source")
    call_id = source.get("callId") if is_json_object(source) else None
    content = message.get("content")
    block = content[0] if isinstance(content, list) and len(content) == 1 else None
    wrapper = block if is_json_object(block) else None
    message_id = message.get("id")
    subject = f"format v3 {event['type']} at seq {event.get('seq')}"
    if (
        not isinstance(message_id, str)
        or not message_id
        or not is_json_object(source)
        or source.get("kind") != "tool"
        or not isinstance(call_id, str)
        or not call_id
        or wrapper is None
        or wrapper.get("type") != "tool-result"
        or wrapper.get("toolCallId") != call_id
    ):
        raise SessionFormatError(
            f"{subject} requires exactly one tool-result wrapper matching "
            "its tool source"
        )
    is_error = wrapper.get("isError")
    if is_error is not None and not isinstance(is_error, bool):
        raise SessionFormatError(f"{subject} tool-result isError must be boolean")
    target_message: JsonObject = {
        "role": "tool",
        "source": source,
        "toolCallId": call_id,
        "content": _result_content(wrapper.get("content"), subject),
        **({} if is_error is None else {"isError": is_error}),
        "id": message_id,
        **_extension_fields(message, _MESSAGE_FIELDS, "message"),
        **_extension_fields(wrapper, _WRAPPER_FIELDS, "result"),
    }
    return {**event, "data": {**data, "message": target_message}}


_V3_BLOCK_TYPES = frozenset(
    ["text", "reasoning", "image", "file", "tool-call", "tool-result"]
)


def _migrate_v3_block(value: Any, subject: str) -> Any:
    if not is_json_object(value) or not isinstance(value.get("type"), str):
        raise SessionFormatError(
            f"{subject} requires content blocks with string type tags"
        )
    block_type = value["type"]
    if block_type in _V3_BLOCK_TYPES:
        return value
    return {**value, "type": f"plugin:{block_type}"}


def _migrate_v3_content(value: Any, subject: str) -> list[Any]:
    if not isinstance(value, list):
        raise SessionFormatError(f"{subject} content must be an array")
    mapped = [
        _migrate_v3_block(block, f"{subject}[{index}]")
        for index, block in enumerate(value)
    ]
    return value if all(mapped[i] is value[i] for i in range(len(mapped))) else mapped


def _migrate_v3_message(message: JsonObject, subject: str) -> JsonObject:
    content = _migrate_v3_content(message.get("content"), subject)
    return (
        message
        if content is message.get("content")
        else {**message, "content": content}
    )


def _migrate_v3_chunk(value: Any, subject: str) -> Any:
    if not is_json_object(value):
        return value
    if value.get("type") == "block-end":
        block = _migrate_v3_block(value.get("block"), f"{subject}.block")
        return value if block is value.get("block") else {**value, "block": block}
    if value.get("type") != "block-start":
        return value
    original = value.get("blockType")
    if not isinstance(original, str):
        raise SessionFormatError(f"{subject} blockType must be a string")
    block_type = original if original in _V3_BLOCK_TYPES else f"plugin:{original}"
    return value if block_type == original else {**value, "blockType": block_type}


def migrate_v3_event_content(event: JsonObject) -> JsonObject:
    """Convert declared v3 extension content and stream tags."""
    subject = f"format v3 {event['type']} at seq {event.get('seq')}"
    mapped = _map_event_messages(
        event, lambda message: _migrate_v3_message(message, subject)
    )
    if not is_json_object(mapped.get("data")):
        return mapped
    data = mapped["data"]
    if event["type"] == "compaction/summary":
        content = _migrate_v3_content(data.get("summary"), f"{subject}.summary")
        if content is not data.get("summary"):
            data = {**data, "summary": content}
        if data.get("rawOutput") is not None:
            content = _migrate_v3_content(data.get("rawOutput"), f"{subject}.rawOutput")
            if content is not data.get("rawOutput"):
                data = {**data, "rawOutput": content}
    elif event["type"] == "tool/ptc-dispatch":
        content = _migrate_v3_content(data.get("content"), f"{subject}.content")
        if content is not data.get("content"):
            data = {**data, "content": content}
    elif event["type"] == "team/message/queued" and is_json_object(data.get("message")):
        message = _migrate_v3_message(data["message"], subject)
        if message is not data["message"]:
            data = {**data, "message": message}
    if event["type"] == "request/header" and is_json_object(data.get("header")):
        tools = data["header"].get("tools")
        if isinstance(tools, list):
            for index, tool in enumerate(tools):
                if is_json_object(tool) and "deferLoading" in tool:
                    raise SessionFormatUnsupportedMigrationError(
                        f"{subject}.header.tools[{index}] contains deferLoading, "
                        "which is only defined in V4"
                    )
    if event["type"] in ("assistant/message", "assistant/attempt") and isinstance(
        data.get("stream"), list
    ):
        stream = data["stream"]
        converted = []
        for index, entry in enumerate(stream):
            if not is_json_object(entry) or entry.get("type") != "chunk":
                converted.append(entry)
                continue
            chunk = _migrate_v3_chunk(entry.get("chunk"), f"{subject}.stream[{index}]")
            converted.append(
                entry if chunk is entry.get("chunk") else {**entry, "chunk": chunk}
            )
        if any(converted[i] is not stream[i] for i in range(len(converted))):
            data = {**data, "stream": converted}
    if data is not mapped["data"]:
        mapped = {**mapped, "data": data}
    return mapped


def historical_child_catalog_source(artifact: SessionFormatArtifact) -> JsonObject:
    """Collect a child's own descriptor for catalog completion."""
    header = artifact.header
    if header.get("origin") != "subagent" or header.get("parentSession") is None:
        raise SessionFormatUnsupportedMigrationError(
            "catalog migration requires a subagent child with a direct parent"
        )
    descriptors = [
        event
        for event in artifact.events
        if event.get("type") == "subagent/descriptor"
        and event.get("seq", 0) >= artifact.inherited_event_count
    ]
    source: JsonObject = {
        "childId": header["id"],
        "childCreatedAt": header["createdAt"],
        "descriptorCount": len(descriptors),
        "descriptor": descriptors[0].get("data") if descriptors else None,
    }
    child_catalog_fact(source)
    return source


def child_catalog_source(value: Any) -> JsonObject:
    """Validate supplemental child evidence supplied across the interface."""
    if (
        not is_json_object(value)
        or not isinstance(value.get("childId"), str)
        or "descriptor" not in value
    ):
        raise SessionFormatUnsupportedMigrationError(
            "catalog migration requires historical child identity and "
            "descriptor evidence"
        )
    require_count(value.get("childCreatedAt"), "catalog child creation time")
    require_count(value.get("descriptorCount"), "child descriptor count")
    return value


def child_catalog_fact(source: JsonObject) -> JsonObject | None:
    """Interpret only the discovery fields of known historical descriptors."""
    child_id = source["childId"]
    descriptor = source.get("descriptor")
    count = source.get("descriptorCount")
    known = is_json_object(descriptor) and descriptor.get("version") in (1, 2, 3)
    if count != 1 or not known:
        return None
    assert is_json_object(descriptor)
    if not isinstance(descriptor.get("provider"), str):
        raise SessionFormatUnsupportedMigrationError(
            f"{child_catalog_subject(source)} has an invalid subagent "
            "descriptor provider"
        )
    if descriptor.get("version") != 1 and descriptor.get("mode") not in (
        "continuable",
        "one-shot",
    ):
        raise SessionFormatUnsupportedMigrationError(
            f"{child_catalog_subject(source)} has an invalid subagent descriptor mode"
        )
    return catalog_fact(
        {
            "version": 0,
            "childId": child_id,
            "childCreatedAt": source["childCreatedAt"],
            "mode": (
                "continuable"
                if descriptor.get("version") == 1
                else descriptor.get("mode")
            ),
            **(
                {}
                if descriptor.get("label") is None
                else {"label": descriptor.get("label")}
            ),
        },
        child_catalog_subject(source),
    )


def catalog_fact(value: Any, subject: str = "subagent/catalog") -> JsonObject:
    """Validate the historical fields used for catalog membership."""
    if (
        not is_json_object(value)
        or value.get("version") not in (0, 1, 2)
        or not isinstance(value.get("childId"), str)
        or value.get("mode") not in ("continuable", "one-shot", "external", "unknown")
        or ((value.get("version") == 2) != (value.get("mode") == "external"))
        or (value.get("version") == 0 and value.get("mode") == "unknown")
        or (
            value.get("mode") == "continuable"
            and not isinstance(value.get("label"), str)
        )
        or (value.get("label") is not None and not isinstance(value.get("label"), str))
    ):
        raise SessionFormatError(
            f"{subject} requires a supported versioned catalog fact"
        )
    require_count(value.get("childCreatedAt"), "catalog child creation time")
    return value


def child_catalog_subject(source: JsonObject) -> str:
    """Name the child and its optional storage-owned location."""
    subject = f"Session {source['childId']}"
    if isinstance(source.get("sourcePath"), str):
        subject += f" (raw log: {source['sourcePath']})"
    return subject


def _rewrite_event_message_source(message: JsonObject, seq: int) -> JsonObject:
    source = message.get("source")
    if not is_json_object(source):
        return message
    converted = rewrite_v3_message_source(source, seq, message.get("role"))
    return message if converted is source else {**message, "source": converted}


class _V3ToV4Stage(_Stage):
    """Convert v3 source events to v4, appending historical child facts."""

    def __init__(
        self,
        input_: SessionFormatMigrationStageInput,
        children: list[Any],
    ) -> None:
        assert_v3_header(input_.source_header)
        self._input = input_
        self._candidates = sorted(
            (child_catalog_source(child) for child in children),
            key=lambda source: (
                source["childCreatedAt"],
                source["childId"],
            ),
        )
        self._catalogs: list[Any] = []
        self._cut: int | None = None if input_.source_header["isSeeded"] else 0
        self._source_cut: int | None = self._cut
        self.header_inherited_event_count = (
            0 if not input_.source_header["isSeeded"] else None
        )
        self._mapping: list[int] = []
        self._turn: int | None = None
        self._step_open = False
        self._next_turn_spliced = False
        self._next_seq = 0
        self._time = input_.source_header["createdAt"]
        self._foreign_delivery_seq: int | None = None

    def _observe_restart(self, event: JsonObject) -> int | None:
        data = event.get("data")
        interrupted = None
        if (
            event.get("type") == "turn/start"
            and self._turn is not None
            and not self._step_open
            and self._next_turn_spliced
            and is_json_object(data)
            and data.get("turn") == self._turn + 1
        ):
            interrupted = self._turn
        self._next_turn_spliced = (
            event.get("type") == "agent/inbox/spliced"
            and is_json_object(data)
            and data.get("target") == "next-turn"
            and isinstance(data.get("inserted"), list)
            and len(data["inserted"]) > 0
        )
        if (
            event.get("type") == "turn/start"
            and is_json_object(data)
            and isinstance(data.get("turn"), int)
            and not isinstance(data.get("turn"), bool)
        ):
            self._turn = data["turn"]
        elif event.get("type") == "turn/end":
            self._turn = None
        elif event.get("type") == "step/start":
            self._step_open = True
        elif event.get("type") == "step/end":
            self._step_open = False
        return interrupted

    def transform_event(self, event: JsonObject, emit: _Emit) -> None:
        if event["seq"] != len(self._mapping):
            raise SessionFormatError("V3 source events must be dense")
        interrupted = self._observe_restart(event)
        if interrupted is not None:
            emit.emit_event(
                {
                    "type": "turn/end",
                    "seq": self._next_seq,
                    "time": event["time"],
                    "data": {"turn": interrupted, "reason": {"kind": "interrupted"}},
                }
            )
            self._next_seq += 1
        target_seq = self._next_seq
        self._next_seq += 1
        self._time = event["time"]
        data = event.get("data")
        if (
            event.get("type") == "session/end-seed"
            and is_json_object(data)
            and data.get("inherited") is True
        ):
            if not self._input.source_header["isSeeded"]:
                raise SessionFormatError(
                    "unseeded format v3 Session contains an inherited end-seed marker"
                )
            self._source_cut = event["seq"]
            self._cut = target_seq
            self._catalogs = []
        elif event.get("type") == "subagent/catalog":
            self._catalogs.append(data)
        delivery_id = validate_delivery_accepted(event, 3)
        if event.get("type") == "session-log-deepseek/delivery-accepted":
            delivery_data = require_record(data, "delivery-accepted data")
            if delivery_data.get("sessionFormatVersion") == 4:
                raise SessionFormatUnsupportedMigrationError(
                    "format v3 delivery marker claims target format v4"
                )
            if (
                delivery_id is not None
                and delivery_id != self._input.source_header["id"]
            ):
                self._foreign_delivery_seq = event["seq"]
        opaque = namespace_v3_opaque_event(event)
        if opaque is not event:
            self._mapping.append(target_seq)
            emit.emit_event(
                opaque if opaque["seq"] == target_seq else {**opaque, "seq": target_seq}
            )
            return
        if event.get("type") not in RELEASED_V3_EVENT_TYPES:
            raise SessionFormatUnsupportedMigrationError(
                "format v3 contains unknown event type "
                f"{json.dumps(event.get('type'))} at seq {event.get('seq')}"
            )
        remapped = _remap_v3_references(event, target_seq, self._mapping)
        self._mapping.append(target_seq)
        rewritten = _map_event_messages(
            remapped,
            lambda message: _rewrite_event_message_source(message, event["seq"]),
        )
        emit.emit_event(migrate_v3_event_content(lift_tool_result(rewritten)))

    def transform_run(self, run: SessionFormatEventRun, emit: _Emit) -> None:
        for event in run.expand():
            self.transform_event(event, emit)

    def finish(self, emit: _Emit) -> int:
        cut = require_count(self._cut, "V3 inherited event count")
        source_cut = require_count(self._source_cut, "V3 source inherited event count")
        existing_catalogs: dict[str, JsonObject] = {}
        for data in self._catalogs:
            fact = catalog_fact(data)
            child_id = fact["childId"]
            if child_id in existing_catalogs:
                raise SessionFormatUnsupportedMigrationError(
                    f"duplicate catalog child {child_id}"
                )
            existing_catalogs[child_id] = fact
        if (
            self._input.source_inherited_event_count is not None
            and source_cut != self._input.source_inherited_event_count
        ):
            raise SessionFormatError(
                "format v3 inherited cut disagrees with its source marker"
            )
        if self._foreign_delivery_seq is not None and (
            self._input.source_header.get("parentSession") is None
            or self._foreign_delivery_seq >= source_cut
        ):
            raise SessionFormatError(
                "current-generation delivery marker names the wrong Session"
            )
        for source in self._candidates:
            child_id = source["childId"]
            existing = existing_catalogs.get(child_id)
            candidate_fact = child_catalog_fact(source)
            subject = child_catalog_subject(source)
            if existing is not None:
                if existing["childCreatedAt"] != source["childCreatedAt"]:
                    raise SessionFormatUnsupportedMigrationError(
                        f"{subject} conflicts with its parent catalog"
                    )
                if candidate_fact is not None and any(
                    existing.get(key) != candidate_fact.get(key)
                    for key in ("mode", "label")
                ):
                    raise SessionFormatUnsupportedMigrationError(
                        f"{subject} conflicts with its parent catalog"
                    )
                continue
            entry = candidate_fact or {
                "version": 1,
                "childId": child_id,
                "childCreatedAt": source["childCreatedAt"],
                "mode": "unknown",
            }
            existing_catalogs[child_id] = entry
            emit.emit_event(
                {
                    "type": "subagent/catalog",
                    "seq": self._next_seq,
                    "time": self._time,
                    "data": entry,
                }
            )
            self._next_seq += 1
        return cut


def _v3_to_v4_create_stage(
    input_: SessionFormatMigrationStageInput,
) -> _Stage:
    raise SessionFormatUnsupportedMigrationError(
        "V3 catalog migration requires explicit historical child facts, "
        "including an empty array for a parent without children"
    )


def _v3_to_v4_migrate_header(header: JsonObject) -> JsonObject:
    assert_v3_header(header)
    return {**header, "version": 4}


SESSION_FORMAT_V3_TO_V4 = SessionFormatMigration(
    name="session-format-v3-to-v4",
    from_version=3,
    to_version=4,
    migrate_header=_v3_to_v4_migrate_header,
    create_stage=_v3_to_v4_create_stage,
    validate_target_header=assert_v4_header,
)


def create_session_format_v3_to_v4(
    children: list[Any],
) -> SessionFormatMigration:
    """Bind one parent's historical child evidence to its v3→v4 migration."""
    return SessionFormatMigration(
        name=SESSION_FORMAT_V3_TO_V4.name,
        from_version=SESSION_FORMAT_V3_TO_V4.from_version,
        to_version=SESSION_FORMAT_V3_TO_V4.to_version,
        migrate_header=SESSION_FORMAT_V3_TO_V4.migrate_header,
        create_stage=lambda input_: _V3ToV4Stage(input_, children),
        validate_target_header=SESSION_FORMAT_V3_TO_V4.validate_target_header,
    )


# ---------------------------------------------------------------------------
# Released artifact restoration (released checks only)
# ---------------------------------------------------------------------------

_V2_EVENT_REQUIRED = ("type", "seq", "time", "data")
_V2_SURFACE_OPTIONAL = ("ignorable", "sourceEventSeqs", "surfaceOp")
_V2_LOG_OPTIONAL = ("ignorable",)


def assert_released_v4_relationships(
    artifact: SessionFormatArtifact, known_event_types: frozenset[str]
) -> None:
    """Validate v4 developer fields, sources, catalog, and delivery ownership.

    Lifecycle relationships (turn/step/tool/retry/title/command agreement)
    belong to installed Session validation and are a P2-1 follow-up.
    """
    ids: set[str] = set()
    for event in artifact.events:
        if event.get("type") not in known_event_types:
            continue
        assert_v4_developer_data(event)
        assert_v4_message_sources(event)
        delivery_id = validate_delivery_accepted(event, 4)
        if (
            delivery_id is not None
            and event.get("seq", 0) >= artifact.inherited_event_count
            and delivery_id != artifact.header.get("id")
        ):
            raise SessionFormatError(
                "current-generation delivery marker names the wrong Session"
            )
        if (
            event.get("type") == "subagent/catalog"
            and event.get("seq", 0) >= artifact.inherited_event_count
        ):
            fact = catalog_fact(event.get("data"))
            child_id = fact["childId"]
            if child_id in ids:
                raise SessionFormatError(f"duplicate catalog child {child_id}")
            ids.add(child_id)


def restore_released_v4_artifact(
    artifact: SessionFormatArtifact, known_event_types: frozenset[str]
) -> SessionFormatArtifact:
    """Validate v4 inheritance, vocabulary, admission, and inherited cut."""
    assert_v4_header(artifact.header)
    cut = require_count(
        artifact.inherited_event_count, "format v4 inherited event count"
    )
    if cut > len(artifact.events):
        raise SessionFormatError("format v4 inherited event count exceeds its events")
    if not artifact.header["isSeeded"] and cut != 0:
        raise SessionFormatError("unseeded format v4 Session has inherited events")
    last_inherited_marker: int | None = None
    for index, event in enumerate(artifact.events):
        if (
            event.get("type") not in known_event_types
            and event.get("ignorable") is not True
        ):
            raise SessionFormatUnsupportedMigrationError(
                "format v4 contains unknown event type "
                f"{json.dumps(event.get('type'))} at seq {index}"
            )
        if event.get("seq") != index:
            raise SessionFormatError(f"format v4 event {index} is not dense")
        if event.get("type") not in known_event_types:
            continue
        assert_v4_retired_syntax(event)
        assert_v4_system_message_fields(event)
        assert_v4_tool_result_message(event)
        assert_v4_fork_result(event)
        if (
            event.get("type") == "session/end-seed"
            and is_json_object(event.get("data"))
            and event["data"].get("inherited") is True
        ):
            last_inherited_marker = index
    if artifact.header["isSeeded"] and last_inherited_marker != cut:
        raise SessionFormatError(
            "format v4 seeded header disagrees with its last inherited end-seed marker"
        )
    if not artifact.header["isSeeded"] and last_inherited_marker is not None:
        raise SessionFormatError(
            "format v4 unseeded Session contains an inherited end-seed marker"
        )
    assert_released_v4_relationships(artifact, known_event_types)
    return artifact


def _restore_released_v2_artifact(
    artifact: SessionFormatArtifact,
    known_event_types: frozenset[str] | None,
) -> SessionFormatArtifact:
    """Validate v2 physical coordinates (header, density, cut, markers).

    Payload semantics and lifecycle relationships are P2-1 follow-ups; this
    validator covers only the frozen physical coordinates.
    """
    assert_v2_header(artifact.header)
    cut = require_count(
        artifact.inherited_event_count, "format v2 inherited event count"
    )
    if cut > len(artifact.events):
        raise SessionFormatError("format v2 inherited event count exceeds its events")
    if not artifact.header["isSeeded"] and cut != 0:
        raise SessionFormatError("unseeded format v2 Session has inherited events")
    last_inherited_marker: int | None = None
    for index, event in enumerate(artifact.events):
        record = require_record(event, f"format v2 event {index}")
        event_type = record.get("type")
        if not isinstance(event_type, str):
            raise SessionFormatError(f"format v2 event {index} type must be a string")
        disposition = RELEASED_V2_EVENT_DISPOSITIONS.get(event_type)
        installed = known_event_types is not None and event_type in known_event_types
        ignorable_unknown = disposition is None and record.get("ignorable") is True
        if disposition is None and not installed and not ignorable_unknown:
            raise SessionFormatUnsupportedMigrationError(
                "format v2 contains unknown event type "
                f"{json.dumps(event_type)} at seq {index}"
            )
        surface = disposition is not None and event_type in SURFACE_TYPES
        optional = (
            _V2_SURFACE_OPTIONAL if disposition is None or surface else _V2_LOG_OPTIONAL
        )
        exact_keys(
            record, list(_V2_EVENT_REQUIRED), list(optional), f"format v2 event {index}"
        )
        if record.get("seq") != index:
            raise SessionFormatError(f"format v2 event {index} is not dense")
        require_safe_integer(record.get("time"), f"format v2 event {index} time")
        if record.get("ignorable") is not None and record["ignorable"] is not True:
            raise SessionFormatError(
                f"format v2 event {index} ignorable must be true when present"
            )
        if event_type == "session/end-seed":
            data = require_record(event.get("data"), f"session/end-seed {index} data")
            if data.get("inherited") is True:
                last_inherited_marker = index
    if artifact.header["isSeeded"] and last_inherited_marker != cut:
        raise SessionFormatError(
            "format v2 seeded header disagrees with its last inherited end-seed marker"
        )
    if not artifact.header["isSeeded"] and last_inherited_marker is not None:
        raise SessionFormatError(
            "format v2 unseeded Session contains an inherited end-seed marker"
        )
    return artifact


def restore_released_v3_artifact(
    artifact: SessionFormatArtifact, known_event_types: frozenset[str]
) -> SessionFormatArtifact:
    """Validate v3 system-head rules, then v2 physical coordinates."""
    assert_v3_header(artifact.header)
    step: dict[str, Any] | None = None
    head: int | None = None
    has_surface = False
    for event in artifact.events:
        assert_v3_event_admission(event)
        assert_v3_event(event, known_event_types)
        system = event.get("type") == "system/message"
        data = require_record(event.get("data"), str(event.get("type")))
        if event.get("type") == "step/start":
            step = {"turn": data["turn"], "step": data["step"]}
        elif event.get("type") in ("step/end", "turn/end"):
            step = None
        if system:
            if (
                step is None
                or step["turn"] != data.get("turn")
                or step["step"] != data.get("step")
            ):
                raise SessionFormatError("system/message does not match an open step")
            operation = event.get("surfaceOp")
            if has_surface and head is None:
                raise SessionFormatError(
                    "system/message requires a protected first surface head"
                )
            if operation == "append":
                if not has_surface:
                    head = event["seq"]
            else:
                replace = require_record(operation, "system replacement")
                if replace.get("startSeq") == head or replace.get("endSeq") == head:
                    if replace.get("startSeq") != head or replace.get("endSeq") != head:
                        raise SessionFormatError(
                            "system/message must replace exactly the current "
                            "system head"
                        )
                    head = event["seq"]
        elif event.get("type") in SURFACE_TYPES and event.get("surfaceOp") != "append":
            replace = require_record(event.get("surfaceOp"), "surface replacement")
            if replace.get("startSeq") == head or replace.get("endSeq") == head:
                raise SessionFormatError(
                    "surface replacement cannot shadow the protected system head"
                )
        if event.get("type") in ("compaction/prune", "compaction/summary"):
            seqs = data.get("shadowedSeqs")
            if isinstance(seqs, list) and head in seqs:
                raise SessionFormatError(
                    "compaction cannot shadow the protected system head"
                )
        if event.get("type") in SURFACE_TYPES:
            has_surface = True
    _restore_released_v2_artifact(
        SessionFormatArtifact(
            header={**artifact.header, "version": 2},
            inherited_event_count=artifact.inherited_event_count,
            events=artifact.events,
        ),
        known_event_types,
    )
    return artifact


# ---------------------------------------------------------------------------
# Catalog: physical codec dispatch + complete adjacent chain
# ---------------------------------------------------------------------------

_SessionFormatCodec = _V01Codec | _V2Codec | _V3Codec | _V4Codec
_SessionFormatEncoder = _V2Codec | _V3Codec | _V4Codec


@dataclass(frozen=True)
class SessionFormatHeaderReadResult:
    """Header-only classification result; never decodes event rows."""

    status: str  # 'current' | 'migration-required' | 'unsupported' | 'malformed'
    target_version: int
    stored_version: int | None = None
    header: JsonObject | None = None
    reason: str | None = None


def _finish_decoder(decoder: _Decoder, source_cut: int | None) -> int:
    inherited = decoder.finish()
    if source_cut is not None and inherited != source_cut:
        raise SessionFormatError(
            "streaming decoder changed its predeclared inherited cut"
        )
    return inherited


def _restore_current_version(
    artifact: SessionFormatArtifact, current_version: int
) -> SessionFormatArtifact:
    if artifact.header.get("version") != current_version:
        raise SessionFormatError(
            "current Session restorer returned v"
            f"{artifact.header.get('version')}; expected v{current_version}"
        )
    return artifact


class _CurrentSessionFormatRestore:
    """Restore for artifacts already at the current version."""

    def __init__(
        self,
        decoder: _Decoder,
        source_cut: int | None,
        restore_artifact: Callable[[SessionFormatArtifact], SessionFormatArtifact],
        current_version: int,
    ) -> None:
        self.header = decoder.header
        self._decoder = decoder
        self._source_cut = source_cut
        self._restore_artifact = restore_artifact
        self._current_version = current_version
        self._collector = _Collector()

    def decode_row(self, row_value: Any) -> None:
        """Decode one physical row into the collector."""
        self._decoder.decode_row(row_value, self._collector)

    def finish(self) -> SessionFormatArtifact:
        """Finish row validation and return the validated current artifact."""
        inherited = _finish_decoder(self._decoder, self._source_cut)
        return _restore_current_version(
            self._restore_artifact(
                SessionFormatArtifact(
                    header=self.header,
                    inherited_event_count=inherited,
                    events=self._collector.values,
                )
            ),
            self._current_version,
        )


class _MigratingSessionFormatRestore(_Emit):
    """Restore for artifacts migrated up to the current version."""

    def __init__(
        self,
        decoder: _Decoder,
        source_cut: int | None,
        migration: _MigrationStream,
        collector: _Collector,
        restore_artifact: Callable[[SessionFormatArtifact], SessionFormatArtifact],
        validation: str,
        source_version: int,
        current_version: int,
    ) -> None:
        self.header = migration.header
        self._decoder = decoder
        self._source_cut = source_cut
        self._migration = migration
        self._collector = collector
        self._restore_artifact = restore_artifact
        self._validation = validation
        self._source_version = source_version
        self._current_version = current_version

    def decode_row(self, row_value: Any) -> None:
        """Decode one physical row through the migration stream."""
        self._decoder.decode_row(row_value, self)

    def emit_event(self, event: JsonObject) -> None:
        """Deliver one decoded event into the migration stream."""
        self._migration.emit_event(event)

    def emit_run(self, run: SessionFormatEventRun) -> None:
        """Deliver one decoded compact run into the migration stream."""
        self._migration.emit_run(run)

    def finish(self) -> SessionFormatArtifact:
        """Finish migration and return the validated current artifact."""
        _finish_decoder(self._decoder, self._source_cut)
        artifact = SessionFormatArtifact(
            header=self.header,
            inherited_event_count=self._migration.finish(),
            events=self._collector.values,
        )
        try:
            restored = self._restore_artifact(artifact)
        except SessionFormatError as error:
            if self._validation == "current" or isinstance(
                error, SessionFormatUnsupportedMigrationError
            ):
                raise
            detail = str(error)
            raise SessionFormatUnsupportedMigrationError(
                f"Session migration from v{self._source_version} to "
                f"v{self._current_version} refuses the transformed artifact: "
                f"{detail}"
            ) from error
        return _restore_current_version(restored, self._current_version)


class SessionFormatCatalog:
    """Build-static physical codec and adjacent migration catalog."""

    def __init__(
        self,
        current_version: int,
        codecs: list[_SessionFormatCodec],
        current_encoder: _SessionFormatEncoder,
        migrations: list[SessionFormatMigration],
        restore_current: Callable[[SessionFormatArtifact], SessionFormatArtifact],
        restore_transformed_current: Callable[
            [SessionFormatArtifact], SessionFormatArtifact
        ],
        restore_current_header: Callable[[JsonObject], JsonObject],
    ) -> None:
        self.current_version = require_format_version(
            current_version, "current Session format version"
        )
        self._chain = SessionFormatChain(
            current_version, migrations, restore_current_header
        )
        self._codecs: dict[int, _SessionFormatCodec] = {}
        for codec in codecs:
            version = require_format_version(
                codec.version, "Session format codec version"
            )
            if version in self._codecs:
                raise SessionFormatError(
                    f"Session format codec v{version} is duplicated"
                )
            self._codecs[version] = codec
        for version in range(self.current_version + 1):
            if version not in self._codecs:
                raise SessionFormatError(f"Session format codec v{version} is missing")
        if len(self._codecs) != self.current_version + 1:
            invalid = next(
                version for version in self._codecs if version > self.current_version
            )
            raise SessionFormatError(
                "Session format codec v"
                f"{invalid} is newer than current v{self.current_version}"
            )
        self._current_encoder = current_encoder
        self._restore_current = restore_current
        self._restore_transformed_current = restore_transformed_current

    def read_header(self, header_value: Any) -> SessionFormatHeaderReadResult:
        """Classify a stored header without decoding any event row."""
        try:
            stored_version = inspect_format_version(header_value)
        except SessionFormatError as error:
            return SessionFormatHeaderReadResult(
                status="malformed",
                target_version=self.current_version,
                reason=str(error),
            )
        if stored_version > self.current_version:
            return SessionFormatHeaderReadResult(
                status="unsupported",
                stored_version=stored_version,
                target_version=self.current_version,
                reason=(
                    f"stored Session uses newer format v{stored_version}; "
                    f"this build writes v{self.current_version}"
                ),
            )
        codec = self._codecs.get(stored_version)
        if codec is None:  # pragma: no cover - construction proves coverage
            return SessionFormatHeaderReadResult(
                status="unsupported",
                stored_version=stored_version,
                target_version=self.current_version,
                reason=(
                    f"this build has no Session format codec for v{stored_version}"
                ),
            )
        try:
            decoded = snapshot_header(
                codec.decode_header(header_value),
                f"format v{stored_version} header",
            )
            header = self._chain.migrate_header(decoded)
            return SessionFormatHeaderReadResult(
                status=(
                    "current"
                    if stored_version == self.current_version
                    else "migration-required"
                ),
                stored_version=stored_version,
                target_version=self.current_version,
                header=header,
            )
        except SessionFormatUnsupportedMigrationError as error:
            return SessionFormatHeaderReadResult(
                status="unsupported",
                stored_version=stored_version,
                target_version=self.current_version,
                reason=str(error),
            )
        except SessionFormatError as error:
            return SessionFormatHeaderReadResult(
                status="malformed",
                stored_version=stored_version,
                target_version=self.current_version,
                reason=str(error),
            )

    def _artifact_codec(self, header_value: Any) -> tuple[int, _SessionFormatCodec]:
        stored_version = inspect_format_version(header_value)
        if stored_version > self.current_version:
            raise SessionFormatUnsupportedMigrationError(
                f"stored Session uses newer format v{stored_version}; "
                f"this build writes v{self.current_version}"
            )
        codec = self._codecs.get(stored_version)
        if codec is None:  # pragma: no cover - construction proves coverage
            raise SessionFormatUnsupportedMigrationError(
                f"this build has no Session format codec for v{stored_version}"
            )
        return stored_version, codec

    def encode_current_header(
        self, header: JsonObject, inherited_event_count: int
    ) -> JsonObject:
        """Encode one current logical header as a physical header line."""
        if inspect_format_version(header) != self.current_version:
            raise SessionFormatError(
                f"encodeCurrent requires Session format v{self.current_version}"
            )
        encoded = self._current_encoder.encode_header(header, inherited_event_count)
        if inspect_format_version(encoded) != self.current_version:
            raise SessionFormatError(
                "current Session codec returned a non-current header"
            )
        return encoded

    def encode_current_event(self, event: JsonObject) -> JsonObject:
        """Encode one current logical event as a physical row."""
        return self._current_encoder.encode_event(event)

    def create_restore(
        self,
        header_value: Any,
        validation: str = "current",
        recovery: str = "strict",
    ) -> _CurrentSessionFormatRestore | _MigratingSessionFormatRestore:
        """Build a streaming restore for one stored artifact.

        ``validation`` is ``'current'`` (full current validation) or
        ``'transformed'`` (released current validation only after
        migration). ``recovery`` is ``'strict'`` (refuse corrupt rows) or
        ``'recoverable'`` (suppress a recoverable suffix).
        """
        stored_version, codec = self._artifact_codec(header_value)
        decoder = codec.create_decoder(header_value, recovery)
        source_cut = decoder.header_inherited_event_count
        if stored_version == self.current_version:
            return _CurrentSessionFormatRestore(
                decoder,
                source_cut,
                self._restore_current
                if validation == "current"
                else lambda artifact: artifact,
                self.current_version,
            )
        collector = _Collector()
        migration = self._chain.create_stream(decoder.header, source_cut, collector)
        return _MigratingSessionFormatRestore(
            decoder,
            source_cut,
            migration,
            collector,
            self._restore_current
            if validation == "current"
            else self._restore_transformed_current,
            validation,
            stored_version,
            self.current_version,
        )


def create_session_format_catalog(
    current_version: int,
    codecs: list[_SessionFormatCodec],
    current_encoder: _SessionFormatEncoder,
    migrations: list[SessionFormatMigration],
    restore_current: Callable[[SessionFormatArtifact], SessionFormatArtifact],
    restore_transformed_current: Callable[
        [SessionFormatArtifact], SessionFormatArtifact
    ],
    restore_current_header: Callable[[JsonObject], JsonObject],
) -> SessionFormatCatalog:
    """Compile a build-static physical codec and adjacent migration catalog."""
    return SessionFormatCatalog(
        current_version=current_version,
        codecs=codecs,
        current_encoder=current_encoder,
        migrations=migrations,
        restore_current=restore_current,
        restore_transformed_current=restore_transformed_current,
        restore_current_header=restore_current_header,
    )


def _restore_current_v4(
    artifact: SessionFormatArtifact,
) -> SessionFormatArtifact:
    # Installed Session validation (fromRestore) is a P2-1 follow-up; the
    # released v4 restore is the complete P2-0 check.
    return restore_released_v4_artifact(artifact, KNOWN_SESSION_EVENT_TYPES)


def _restore_current_v3(
    artifact: SessionFormatArtifact,
) -> SessionFormatArtifact:
    return restore_released_v3_artifact(artifact, RELEASED_V3_EVENT_TYPES)


def _shipped_codecs() -> list[_SessionFormatCodec]:
    return [_V01Codec(0), _V01Codec(1), _V2Codec(), _V3Codec(), _V4Codec()]


def _shipped_migrations() -> list[SessionFormatMigration]:
    return [
        SESSION_FORMAT_V0_TO_V1,
        SESSION_FORMAT_V1_TO_V2,
        SESSION_FORMAT_V2_TO_V3,
        SESSION_FORMAT_V3_TO_V4,
    ]


#: Physical codec dispatch and complete adjacent chain (current v4).
SESSION_FORMAT_CATALOG = create_session_format_catalog(
    current_version=CURRENT_FORMAT_VERSION,
    codecs=_shipped_codecs(),
    current_encoder=_V4Codec(),
    migrations=_shipped_migrations(),
    restore_current=_restore_current_v4,
    restore_transformed_current=_restore_current_v4,
    restore_current_header=assert_v4_header,
)

#: V0–V3 decoding for historical child identity; never completes parent catalogs.
HISTORICAL_SESSION_FORMAT_CATALOG = create_session_format_catalog(
    current_version=3,
    codecs=[_V01Codec(0), _V01Codec(1), _V2Codec(), _V3Codec()],
    current_encoder=_V3Codec(),
    migrations=[
        SESSION_FORMAT_V0_TO_V1,
        SESSION_FORMAT_V1_TO_V2,
        SESSION_FORMAT_V2_TO_V3,
    ],
    restore_current=_restore_current_v3,
    restore_transformed_current=_restore_current_v3,
    restore_current_header=assert_v3_header,
)


def create_session_format_catalog_with_children(
    children: list[Any],
) -> SessionFormatCatalog:
    """Bind one parent's historical child evidence to the shipped catalog."""
    return create_session_format_catalog(
        current_version=CURRENT_FORMAT_VERSION,
        codecs=_shipped_codecs(),
        current_encoder=_V4Codec(),
        migrations=[
            SESSION_FORMAT_V0_TO_V1,
            SESSION_FORMAT_V1_TO_V2,
            SESSION_FORMAT_V2_TO_V3,
            create_session_format_v3_to_v4(children),
        ],
        restore_current=_restore_current_v4,
        restore_transformed_current=_restore_current_v4,
        restore_current_header=assert_v4_header,
    )


__all__ = [
    "CURRENT_FORMAT_VERSION",
    "HISTORICAL_SESSION_FORMAT_CATALOG",
    "KNOWN_SESSION_EVENT_TYPES",
    "RELEASED_V0_EVENT_DISPOSITIONS",
    "RELEASED_V2_EVENT_DISPOSITIONS",
    "RELEASED_V3_EVENT_TYPES",
    "SESSION_FORMAT_CATALOG",
    "SESSION_FORMAT_V0_TO_V1",
    "SESSION_FORMAT_V1_TO_V2",
    "SESSION_FORMAT_V2_TO_V3",
    "SESSION_FORMAT_V3_TO_V4",
    "SessionFormatArtifact",
    "SessionFormatCatalog",
    "SessionFormatChain",
    "SessionFormatError",
    "SessionFormatEventRun",
    "SessionFormatHeaderReadResult",
    "SessionFormatMigration",
    "SessionFormatMigrationStageInput",
    "SessionFormatUnsupportedError",
    "SessionFormatUnsupportedMigrationError",
    "SessionPersistenceCorruptionError",
    "accepted_through",
    "assert_released_event_payload",
    "assert_released_v4_relationships",
    "assert_v3_event",
    "assert_v3_event_admission",
    "assert_v4_header",
    "assert_v4_row_admission",
    "catalog_fact",
    "child_catalog_fact",
    "child_catalog_source",
    "child_catalog_subject",
    "create_session_format_catalog",
    "create_session_format_catalog_with_children",
    "create_session_format_v3_to_v4",
    "decode_envelope_row",
    "decode_seq_ranges",
    "define_session_format_migration",
    "encode_envelope_event",
    "encode_seq_ranges",
    "historical_child_catalog_source",
    "inspect_format_version",
    "lift_tool_result",
    "migrate_v3_event_content",
    "namespace_v3_opaque_event",
    "parse_session_format_log_filename",
    "restore_released_v3_artifact",
    "restore_released_v4_artifact",
    "rewrite_v3_message_source",
    "session_format_log_filename",
    "session_format_version_refusal",
    "snapshot_header",
    "validate_delivery_accepted",
]
