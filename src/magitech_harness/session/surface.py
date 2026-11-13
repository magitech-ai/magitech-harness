"""Surface fold over a session event log.

The surface is the ordered view of events that produce model-visible
messages; the append-only log remains the source of truth. Only the five
surface types join it. Each surface event carries a surface operation
(``append`` or a ``replace`` over an inclusive seq range citing every
shadowed node); non-surface events must carry neither a surface
operation nor source seqs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeGuard

from magitech_harness.session.events import (
    KNOWN_SESSION_EVENT_TYPES,
    MESSAGE_PROJECTION_EVENT_TYPES,
    SURFACE_EVENT_TYPES,
    SessionEvent,
)
from magitech_harness.session.request_header import REQUEST_HEADER_REASONS
from magitech_harness.session.tool_history import tool_name


class SurfaceError(ValueError):
    """A session event violates the surface contract or payload rules."""


class MessageProjection(Protocol):
    """Pure interpretation of one plugin-owned event that changes messages."""

    type: str

    def project(
        self, event: SessionEvent, context: MessageProjectionContext
    ) -> dict[int, dict[str, Any]]: ...


@dataclass
class MessageProjectionContext:
    """Read-only history immediately before a message-projection event."""

    nodes: list[int]
    events: list[SessionEvent]
    base_seq: int
    messages: dict[int, dict[str, Any]]


@dataclass
class SurfaceFoldReplacement:
    """One replacement operation observed while folding a surface."""

    seq: int
    start: int
    end: int
    shadowed_seqs: list[int]


@dataclass
class SurfaceFoldResult:
    """Complete result of replaying the surface operations in a log."""

    nodes: list[int]
    replacements: list[SurfaceFoldReplacement] = field(default_factory=list)
    projected_messages: dict[int, dict[str, Any]] = field(default_factory=dict)


def _is_record(value: Any) -> TypeGuard[dict[str, Any]]:
    return isinstance(value, dict)


def _is_event_seq(value: Any) -> bool:
    return bool(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and value <= 2**53 - 1
    )


def _is_replace_op(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and len(value) == 3
        and value.get("op") == "replace"
        and _is_event_seq(value.get("startSeq"))
        and _is_event_seq(value.get("endSeq"))
    )


def _content_blocks(message: Any) -> list[Any]:
    if _is_record(message) and isinstance(message.get("content"), list):
        return list(message["content"])
    return []


def validate_session_event_data(
    event: SessionEvent, subject: str, *, strict: bool = False
) -> None:
    """Reject noncanonical payload fields for one event.

    Covers request-header empty-field and retired ``header.system``
    rules, developer-role/tool-change rules, the developer ``source.kind
    === 'plugin'`` refusal, and tool-error consistency. Does not
    validate complete event payloads or embedded provider streams.

    ``strict`` gates the checks that only a fresh write must satisfy:
    an assistant message's model-source attribution (kind, provider,
    model), and the refusal of a legacy husk header. Restored logs
    predate that attribution on assistant sources, so the seed/restore
    path leaves it unenforced; the load layer enforces it for file
    rows, and :meth:`Session.append` always validates strictly.
    """
    data = event.payload
    if event.type in SURFACE_EVENT_TYPES and _is_record(data):
        message = data if event.type == "user/message" else data.get("message")
        if _is_record(message):
            if (event.type == "developer/message") != (
                message.get("role") == "developer"
            ):
                raise SurfaceError(
                    f"{subject} developer/message and developer role must occur "
                    "together"
                )
            blocks = _content_blocks(message)
            if message.get("role") != "developer" and any(
                _is_record(block)
                and block.get("type") in ("tool-addition", "tool-removal")
                for block in blocks
            ):
                raise SurfaceError(
                    f"{subject} tool-change blocks require developer role"
                )
            if event.type == "developer/message":
                source = message.get("source")
                if _is_record(source) and source.get("kind") == "plugin":
                    raise SurfaceError(
                        f"{subject} developer/message source must not be plugin-owned"
                    )
                has_additions = False
                for block in blocks:
                    if not _is_record(block) or block.get("type") not in (
                        "tool-addition",
                        "tool-removal",
                    ):
                        continue
                    if not isinstance(block.get("toolName"), str) or not block.get(
                        "toolName"
                    ):
                        raise SurfaceError(
                            f"{subject} {block.get('type')} requires a nonempty toolName"
                        )
                    if block.get("type") == "tool-addition":
                        has_additions = True
                        if "tool" in block:
                            raise SurfaceError(
                                f"{subject} tool-addition must omit inline tool "
                                "definitions"
                            )
                if has_additions != ("headerSeq" in data):
                    raise SurfaceError(
                        f"{subject} requires headerSeq exactly when tool additions "
                        "are present"
                    )
            if event.type == "system/message":
                if message.get("role") not in (None, "system"):
                    raise SurfaceError(f"{subject} system/message role must be system")
                if any(
                    _is_record(block) and block.get("type") == "tool-result"
                    for block in blocks
                ):
                    raise SurfaceError(
                        f"{subject} system/message must not carry tool-result blocks"
                    )
    if event.type == "request/header":
        if not _is_record(data):
            # Payload is always a dict: pydantic enforces it at
            # SessionEvent construction.
            raise SurfaceError(  # pragma: no cover
                f"{subject} data must be an object"
            )
        header = data.get("header")
        if not _is_record(header):
            raise SurfaceError(f"{subject} header must be an object")
        if "system" in header:
            raise SurfaceError(f"{subject} must omit header.system; use system/message")
        if isinstance(header.get("tools"), list) and len(header["tools"]) == 0:
            raise SurfaceError(f"{subject} must omit empty tools")
        defaults = header.get("adapterDefaults")
        if _is_record(defaults) and len(defaults) == 0:
            raise SurfaceError(f"{subject} must omit empty adapterDefaults")
        # The migration's own husk header (empty header, legacy start
        # reason) carries no config to validate; it is grandfathered
        # here exactly as in the load layer — but only on the
        # restore/seed path. A fresh append (strict) must not write one.
        if not strict and header == {} and data.get("reason") == {"kind": "start"}:
            return
        config = header.get("config")
        if (
            not _is_record(config)
            or not isinstance(config.get("provider"), str)
            or not config["provider"]
            or not isinstance(config.get("model"), str)
            or not config["model"]
        ):
            raise SurfaceError(
                f"{subject} header config requires a nonempty provider and model"
            )
        reasoning_effort = config.get("reasoningEffort")
        if reasoning_effort is not None and (
            not isinstance(reasoning_effort, str) or not reasoning_effort
        ):
            raise SurfaceError(
                f"{subject} header config reasoningEffort must be a nonempty string"
            )
        if (
            not isinstance(data.get("reason"), str)
            or data.get("reason") not in REQUEST_HEADER_REASONS
        ):
            raise SurfaceError(
                f"{subject} reason must be one of {sorted(REQUEST_HEADER_REASONS)}"
            )
        if "startsSeries" in data and data["startsSeries"] is not True:
            raise SurfaceError(f"{subject} startsSeries must be true when present")
        if defaults is not None:
            if not _is_record(defaults):
                raise SurfaceError(f"{subject} adapterDefaults must be an object")
            for key in defaults:
                if key not in ("reasoningEffort", "maxTokens"):
                    raise SurfaceError(
                        f"{subject} adapterDefaults carries unknown key {key!r}"
                    )
            for marker in ("reasoningEffort", "maxTokens"):
                if marker in defaults and defaults[marker] is not True:
                    raise SurfaceError(
                        f"{subject} adapterDefaults {marker} must be true when present"
                    )
                if defaults.get(marker) is True and marker not in config:
                    raise SurfaceError(
                        f"{subject} adapterDefaults {marker} requires config {marker}"
                    )
        tools = header.get("tools")
        if tools is not None:
            if not isinstance(tools, list):
                raise SurfaceError(f"{subject} header tools must be an array")
            for tool in tools:
                if not _is_record(tool):
                    raise SurfaceError(f"{subject} header tools must be objects")
                if "deferLoading" in tool and tool["deferLoading"] is not True:
                    raise SurfaceError(
                        f"{subject} header tool deferLoading must be true when present"
                    )
    if event.type == "request/context":
        if (
            not isinstance(data.get("provider"), str)
            or not data["provider"]
            or not isinstance(data.get("model"), str)
            or not data["model"]
        ):
            raise SurfaceError(
                f"{subject} request/context requires a nonempty provider and model"
            )
        context_window = data.get("contextWindow")
        if context_window is not None and (
            isinstance(context_window, bool)
            or not isinstance(context_window, int)
            or context_window <= 0
            or context_window > 2**53 - 1
        ):
            raise SurfaceError(
                f"{subject} request/context contextWindow must be a positive "
                "safe integer when present"
            )
        if "systemPromptUpdate" in data and data["systemPromptUpdate"] != "in-history":
            raise SurfaceError(
                f"{subject} request/context systemPromptUpdate must be "
                '"in-history" when present'
            )
    if event.type in ("assistant/message", "assistant/attempt"):
        for field in ("turn", "step"):
            if not _is_event_seq(data.get(field)):
                raise SurfaceError(
                    f"{subject} {field} must be a non-negative safe integer"
                )
        if not isinstance(data.get("stream"), list):
            raise SurfaceError(f"{subject} stream must be an array")
        if event.type == "assistant/message":
            if "tool_calls" in data:
                raise SurfaceError(
                    f"{subject} assistant/message must not carry tool_calls"
                )
            if "interrupted" in data and data["interrupted"] is not True:
                raise SurfaceError(f"{subject} interrupted must be true when present")
            message = data.get("message")
            source = message.get("source") if _is_record(message) else None
            if (
                not _is_record(message)
                or not isinstance(message.get("id"), str)
                or not message["id"]
                or message.get("role") != "assistant"
                or not isinstance(message.get("content"), list)
            ):
                raise SurfaceError(
                    f"{subject} assistant/message requires an identified model message"
                )
            # Source attribution is a fresh-write requirement only:
            # restored logs predate it (the golden fixtures and the
            # migration both carry sourceless or unattributed assistant
            # messages), and file loads enforce it in the load layer.
            if strict and (
                not _is_record(source)
                or source.get("kind") != "model"
                or not isinstance(source.get("provider"), str)
                or not source["provider"]
                or not isinstance(source.get("model"), str)
                or not source["model"]
            ):
                raise SurfaceError(
                    f"{subject} assistant/message requires a model source "
                    "with provider and model"
                )
    elif event.type == "tool/result":
        if not _is_record(data):
            # Payload is always a dict: pydantic enforces it at
            # SessionEvent construction.
            raise SurfaceError(  # pragma: no cover
                f"{subject} data must be an object"
            )
        if data.get("error") is None:
            return
        message = data.get("message")
        if not _is_record(message) or message.get("isError") is not True:
            raise SurfaceError(f"{subject} error requires message.isError === true")


def surface_op_of(event: SessionEvent) -> str | dict[str, Any] | None:
    """Validate event-local surface eligibility and return its operation.

    Returns ``None`` for a log-only event or an unknown ignorable record
    (which retains opaque metadata and never changes the surface).
    """
    if event.type not in SURFACE_EVENT_TYPES:
        if event.type not in KNOWN_SESSION_EVENT_TYPES and event.ignorable is True:
            return None
        if event.surface_op is not None:
            raise SurfaceError(
                f'session event "{event.type}" is not surface-eligible and '
                "cannot carry surfaceOp"
            )
        if event.source_event_seqs is not None:
            raise SurfaceError(
                f'session event "{event.type}" is not surface-eligible and '
                "cannot carry sourceEventSeqs"
            )
        return None
    operation = event.surface_op
    if operation is None:
        raise SurfaceError(
            f'session event "{event.type}" is surface-eligible and requires '
            "a surfaceOp marker"
        )
    if operation == "append":
        return operation
    if not _is_replace_op(operation):
        raise SurfaceError(f'session event "{event.type}" carries an invalid surfaceOp')
    return operation


def validate_surface_metadata(event: SessionEvent) -> str | dict[str, Any] | None:
    """Validate one event's surface metadata without checking log membership."""
    operation = surface_op_of(event)
    if (
        operation is not None
        and operation != "append"
        and isinstance(operation, dict)
        and (operation["startSeq"] >= event.seq or operation["endSeq"] >= event.seq)
    ):
        raise SurfaceError(
            f"surface replace at seq {event.seq}: startSeq and endSeq must "
            "reference earlier events"
        )
    if operation is not None:
        _assert_source_event_references(event, [])
    return operation


def _assert_source_event_references(
    event: SessionEvent, shadowed_seqs: list[int]
) -> None:
    if event.type == "assistant/message" and event.source_event_seqs is not None:
        raise SurfaceError(
            "assistant/message embeds its source stream and cannot carry "
            "sourceEventSeqs"
        )
    sources = event.source_event_seqs
    if sources is not None:
        if not isinstance(sources, list):
            # source_event_seqs is list[int] | None: pydantic enforces it
            # at SessionEvent construction.
            raise SurfaceError(  # pragma: no cover
                f"sourceEventSeqs on event at seq {event.seq} must be an array "
                "when present"
            )
        if len(sources) == 0:
            raise SurfaceError("sourceEventSeqs must not be empty")
        seen: set[int] = set()
        for source in sources:
            if not _is_event_seq(source):
                raise SurfaceError(
                    f'session event "{event.type}" sourceEventSeqs must densely '
                    "contain non-negative safe integers"
                )
            seen.add(source)
            if source >= event.seq:
                raise SurfaceError(
                    "sourceEventSeqs must reference earlier events: "
                    f"{source} >= current seq {event.seq}"
                )
        if len(seen) != len(sources):
            raise SurfaceError("sourceEventSeqs must not contain duplicates")
    missing = [seq for seq in shadowed_seqs if seq not in (sources or [])]
    if missing:
        raise SurfaceError(
            "surface replace: sourceEventSeqs must include every shadowed "
            f"surface node; missing {', '.join(str(seq) for seq in missing)}"
        )


def _complete_tool_definition(schema: Any) -> bool:
    """Whether one envelope tool schema carries a complete definition.

    Header tools are envelope schemas; the definition lives on the
    nested function object, not at the top level.
    """
    function = schema.get("function") if isinstance(schema, dict) else None
    if not isinstance(function, dict):
        return False
    return isinstance(function.get("description"), str) and isinstance(
        function.get("parameters"), dict
    )


def _envelope_tool_name(tool: Any) -> str | None:
    """The name of one envelope tool schema, or ``None`` when unnamed."""
    if not isinstance(tool, dict):
        return None
    try:
        return tool_name(tool)
    except ValueError:
        return None


def _assert_developer_header(
    event: SessionEvent, events: list[SessionEvent], base_seq: int
) -> None:
    if event.type != "developer/message":
        return
    validate_session_event_data(event, f"developer/message at seq {event.seq}")
    header_seq = event.payload.get("headerSeq")
    if header_seq is None:
        return
    if not _is_event_seq(header_seq) or header_seq >= event.seq:
        raise SurfaceError(
            "developer/message headerSeq must reference an earlier request/header"
        )
    header_event = events[header_seq - base_seq] if header_seq - base_seq >= 0 else None
    if header_event is None or header_event.type != "request/header":
        raise SurfaceError(
            "developer/message headerSeq must reference an earlier request/header"
        )
    header = header_event.payload.get("header")
    if not _is_record(header):
        raise SurfaceError(
            "developer/message headerSeq must reference a request/header "
            "with a record header"
        )
    tools = header.get("tools")
    for block in _content_blocks(event.payload.get("message")):
        if not _is_record(block) or block.get("type") != "tool-addition":
            continue
        definitions = (
            [
                tool
                for tool in tools
                if _envelope_tool_name(tool) == block.get("toolName")
            ]
            if isinstance(tools, list)
            else []
        )
        if len(definitions) != 1:
            raise SurfaceError(
                f'developer/message tool-addition "{block.get("toolName")}" must '
                f"name exactly one tool in headerSeq {header_seq}"
            )
        definition = definitions[0]
        if not _complete_tool_definition(definition):
            raise SurfaceError(
                f'developer/message tool-addition "{block.get("toolName")}" '
                f"requires a complete tool definition in headerSeq {header_seq}"
            )
        for holder in (definition, definition.get("function")):
            if (
                isinstance(holder, dict)
                and "deferLoading" in holder
                and holder["deferLoading"] is not True
            ):
                raise SurfaceError(
                    "developer/message referenced tool deferLoading must be true "
                    "when present"
                )


def _deep_equal_json(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_deep_equal_json(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return bool(
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_deep_equal_json(left[key], right[key]) for key in left)
        )
    return bool(left == right)


def derive_event_message(
    event: SessionEvent,
    projected_messages: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Project one event into the message it derives to, or ``None``.

    A non-surface event, or an empty-content system, developer, or
    assistant message, produces no message. An empty assistant event
    hosts only usage for a max-tokens step and must not inject a
    content-less turn.
    """
    if projected_messages is not None and event.seq in projected_messages:
        return projected_messages[event.seq]
    if event.type == "user/message":
        return event.payload
    if event.type in ("system/message", "developer/message", "assistant/message"):
        message = event.payload.get("message")
        if not _is_record(message):
            return None
        content = message.get("content")
        if isinstance(content, list) and len(content) == 0:
            return None
        return message
    if event.type == "tool/result":
        message = event.payload.get("message")
        return message if _is_record(message) else None
    return None


@dataclass
class _FoldState:
    nodes: list[int] = field(default_factory=list)
    replace_generation: int = 0
    content_generation: int = 0
    projected_messages: dict[int, dict[str, Any]] = field(default_factory=dict)
    projections: set[Any] = field(default_factory=set)


@dataclass
class _Plan:
    kind: str
    seq: int = 0
    start: int = 0
    end: int = 0
    start_idx: int = 0
    end_idx: int = 0
    shadowed_seqs: list[int] = field(default_factory=list)
    projection: Any = None
    messages: dict[int, dict[str, Any]] = field(default_factory=dict)


def _replacement_range(
    state: _FoldState, operation: dict[str, Any]
) -> tuple[int, int, list[int]]:
    start_idx = (
        state.nodes.index(operation["startSeq"])
        if operation["startSeq"] in state.nodes
        else -1
    )
    if start_idx == -1:
        raise SurfaceError(
            f"surface replace: start seq {operation['startSeq']} not found in surface"
        )
    end_idx = (
        state.nodes.index(operation["endSeq"])
        if operation["endSeq"] in state.nodes
        else -1
    )
    if end_idx == -1:
        raise SurfaceError(
            f"surface replace: end seq {operation['endSeq']} not found in surface"
        )
    if start_idx > end_idx:
        raise SurfaceError(
            f"surface replace: start seq {operation['startSeq']} (index {start_idx}) "
            f"is after end seq {operation['endSeq']} (index {end_idx})"
        )
    return start_idx, end_idx, state.nodes[start_idx : end_idx + 1]


def _assert_tool_result_rewrite(
    event: SessionEvent,
    shadowed_seqs: list[int],
    events: list[SessionEvent],
    base_seq: int,
) -> None:
    if event.type != "tool/result":
        return
    if len(shadowed_seqs) != 1:
        raise SurfaceError(
            "tool/result surface replacement must rewrite exactly one current node"
        )
    original = events[shadowed_seqs[0] - base_seq]
    if original.type != "tool/result":
        raise SurfaceError(
            "tool/result surface replacement must target a current tool/result"
        )
    original_rest = dict(original.payload)
    replacement_rest = dict(event.payload)
    original_message = dict(original.payload.get("message") or {})
    replacement_message = dict(event.payload.get("message") or {})
    original_message["content"] = None
    replacement_message["content"] = None
    original_rest["message"] = original_message
    replacement_rest["message"] = replacement_message
    if not _deep_equal_json(original_rest, replacement_rest):
        raise SurfaceError("tool/result surface replacement may change only content")


def _assert_system_head_rewrite(
    event: SessionEvent,
    state: _FoldState,
    start_idx: int,
    shadowed_seqs: list[int],
    events: list[SessionEvent],
    base_seq: int,
) -> None:
    if start_idx != 0:
        return
    head = events[state.nodes[0] - base_seq]
    if head.type != "system/message":
        return
    if event.type != "system/message" or len(shadowed_seqs) != 1:
        raise SurfaceError(
            "surface replace: node 0 holds the system prompt and may be "
            "rewritten only by a system/message over exactly that node"
        )


def _plan_surface_event(
    state: _FoldState,
    event: SessionEvent,
    expected_seq: int,
    events: list[SessionEvent],
    base_seq: int,
    projections: list[MessageProjection],
) -> _Plan | None:
    if event.seq != expected_seq:
        raise SurfaceError(
            f"session event seq {event.seq} is not contiguous; expected {expected_seq}"
        )
    operation = validate_surface_metadata(event)
    _assert_developer_header(event, events, base_seq)
    projection = next((item for item in projections if item.type == event.type), None)
    if projection is not None:
        messages = projection.project(
            event,
            MessageProjectionContext(
                nodes=list(state.nodes),
                events=list(events),
                base_seq=base_seq,
                messages=dict(state.projected_messages),
            ),
        )
        return _Plan(
            kind="project",
            seq=event.seq,
            projection=projection,
            messages=dict(messages),
        )
    if event.type in MESSAGE_PROJECTION_EVENT_TYPES:
        raise SurfaceError(
            f'session event "{event.type}" requires a message projection; load '
            "its owning plugin or supply its projection definition"
        )
    if operation is None:
        return None
    if operation == "append":
        return _Plan(kind="append", seq=event.seq)
    assert isinstance(operation, dict)
    start_idx, end_idx, shadowed = _replacement_range(state, operation)
    _assert_source_event_references(event, shadowed)
    _assert_tool_result_rewrite(event, shadowed, events, base_seq)
    _assert_system_head_rewrite(event, state, start_idx, shadowed, events, base_seq)
    return _Plan(
        kind="replace",
        seq=event.seq,
        start=operation["startSeq"],
        end=operation["endSeq"],
        start_idx=start_idx,
        end_idx=end_idx,
        shadowed_seqs=shadowed,
    )


def _apply_plan(state: _FoldState, plan: _Plan | None) -> SurfaceFoldReplacement | None:
    if plan is None:
        return None
    if plan.kind == "append":
        state.nodes.append(plan.seq)
    elif plan.kind == "replace":
        state.nodes[plan.start_idx : plan.end_idx + 1] = [plan.seq]
        state.replace_generation += 1
        state.content_generation += 1
    elif plan.kind == "project":  # pragma: no cover - plan kinds are
        # exactly append / replace / project, and the elif chain above
        # already excluded the first two, so this condition is always true.
        state.projected_messages.update(plan.messages)
        state.projections.add(plan.projection)
        state.content_generation += 1
    if plan.kind != "replace":
        return None
    return SurfaceFoldReplacement(
        seq=plan.seq, start=plan.start, end=plan.end, shadowed_seqs=plan.shadowed_seqs
    )


def fold_surface(
    events: list[SessionEvent],
    projections: list[MessageProjection] | None = None,
) -> SurfaceFoldResult:
    """Replay a complete session log through the canonical surface fold."""
    state = _FoldState()
    replacements: list[SurfaceFoldReplacement] = []
    for index, event in enumerate(events):
        plan = _plan_surface_event(
            state, event, index, events, 0, list(projections or [])
        )
        replacement = _apply_plan(state, plan)
        if replacement is not None:
            replacements.append(replacement)
    return SurfaceFoldResult(
        nodes=list(state.nodes),
        replacements=replacements,
        projected_messages=dict(state.projected_messages),
    )


class SurfaceManager:
    """Incremental ordered surface view and append-boundary validator."""

    def __init__(
        self,
        log: list[SessionEvent],
        projections: list[MessageProjection] | None = None,
        base_seq: int = 0,
    ) -> None:
        self._log = log
        self._base_seq = base_seq
        # Held by reference, not copied: removing a definition from the
        # store's live list must invalidate sessions that used it.
        self._projections: list[MessageProjection] = (
            projections if projections is not None else []
        )
        self._state = _FoldState()
        self._last_processed_seq = base_seq - 1
        self._pending: tuple[SessionEvent, int, _Plan | None] | None = None

    def validate_next(self, event: SessionEvent) -> None:
        """Validate the next candidate without mutating the committed surface."""
        self._assert_projections()
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()
        expected_seq = self._base_seq + len(self._log)
        plan = _plan_surface_event(
            self._state,
            event,
            expected_seq,
            self._log,
            self._base_seq,
            self._projections,
        )
        self._pending = (event, expected_seq, plan)

    @property
    def replace_generation(self) -> int:
        self._assert_projections()
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()
        return self._state.replace_generation

    @property
    def content_generation(self) -> int:
        self._assert_projections()
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()
        return self._state.content_generation

    @property
    def nodes(self) -> list[int]:
        self._assert_projections()
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()
        return list(self._state.nodes)

    def derive_event_message(self, event: SessionEvent) -> dict[str, Any] | None:
        self._assert_projections()
        if self._last_processed_seq < self._base_seq + len(self._log) - 1:
            self._process_delta()
        return derive_event_message(event, self._state.projected_messages)

    def _process_delta(self) -> None:
        tail_seq = self._base_seq + len(self._log) - 1
        seq = self._last_processed_seq + 1
        while seq <= tail_seq:
            index = seq - self._base_seq
            event = self._log[index]
            pending = self._pending
            if pending is not None and pending[0] is event and pending[1] == seq:
                _apply_plan(self._state, pending[2])
            else:
                plan = _plan_surface_event(
                    self._state,
                    event,
                    seq,
                    self._log,
                    self._base_seq,
                    self._projections,
                )
                _apply_plan(self._state, plan)
            if pending is not None and pending[1] <= seq:
                self._pending = None
            self._last_processed_seq = seq
            seq += 1

    def _assert_projections(self) -> None:
        pending = self._pending
        required = set(self._state.projections)
        if (
            pending is not None
            and pending[2] is not None
            and pending[2].kind == "project"
        ):
            required.add(pending[2].projection)
        for projection in required:
            if projection not in self._projections:
                raise SurfaceError(
                    f'session message projection "{projection.type}" was removed '
                    "or replaced; restore the session with its owning plugin"
                )
