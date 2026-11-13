"""The five projection units: pure synchronous folds over committed events.

Each unit is a :class:`ProjectionDefinition`: a key, a state schema that
validates persisted state before it seeds a fold, an ``init`` over the
empty log and its immutable session metadata, a pure ``apply``, an
optional wire view, and a state version. A unit uninterested in an event
returns the same state reference; a unit that needs a payload field the
event does not carry also returns the state unchanged, so the drive stays
total over every admitted event.
"""

from __future__ import annotations

import math
import os
import re
from typing import Annotated, Any, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    WireView,
)

# ---------------------------------------------------------------------------
# Title-text cleaning (escape sequences, controls, whitespace collapse).
# ---------------------------------------------------------------------------

_OSC_SEQUENCE = re.compile(
    r"(?:\x1b\]|\x9d)(?:(?!\x07|\x1b\\)[\s\S])*(?:\x07|\x1b\\|\Z)"
)
_CSI_SEQUENCE = re.compile(r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]")
_ESC_SEQUENCE = re.compile(r"\x1b[@-_]")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_DIRECTIONAL_CONTROL = re.compile(
    r"[\u200b\u200e\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff]"
)
_WHITESPACE_RUN = re.compile(r"\s+")


def _clean_title_text(text: str) -> str:
    """Strip escapes and controls; collapse whitespace runs; trim."""
    cleaned = _OSC_SEQUENCE.sub("", text)
    cleaned = _CSI_SEQUENCE.sub("", cleaned)
    cleaned = _ESC_SEQUENCE.sub("", cleaned)
    cleaned = _CONTROL_CHARACTER.sub("", cleaned)
    cleaned = _DIRECTIONAL_CONTROL.sub("", cleaned)
    return _WHITESPACE_RUN.sub(" ", cleaned).strip()


def _eligible_title_message(event: SessionEvent) -> dict[str, Any] | None:
    """The eligible human message of one event, or ``None``.

    Eligible iff the event is a ``user/message`` whose source kind is
    ``"user"`` and whose joined text blocks clean to non-empty. The
    value carries the raw joined block text; cleaning is only the
    emptiness test.
    """
    if event.type != "user/message":
        return None
    source = event.payload.get("source")
    if not isinstance(source, dict) or source.get("kind") != "user":
        return None
    content = event.payload.get("content")
    if not isinstance(content, list):
        return None
    text = "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )
    if _clean_title_text(text) == "":
        return None
    return {"seq": event.seq, "text": text}


# ---------------------------------------------------------------------------
# First-token time and output tokens (sessionStats helpers).
# ---------------------------------------------------------------------------


def _is_token_delta(chunk: Any) -> bool:
    """Whether one stream chunk carries the model's first output token."""
    if not isinstance(chunk, dict):
        return False
    chunk_type = chunk.get("type")
    if chunk_type in ("text-delta", "reasoning-delta"):
        text = chunk.get("text", "")
        return isinstance(text, str) and text != ""
    if chunk_type == "tool-call-delta":
        args = chunk.get("argumentsDelta", "")
        return (isinstance(args, str) and args != "") or "name" in chunk
    return False


def _record_time0(record: dict[str, Any]) -> float | None:
    """One packed run's base time in ms, or ``None`` when it is not a number."""
    time0 = record.get("time0")
    if isinstance(time0, bool) or not isinstance(time0, (int, float)):
        return None
    return float(time0)


def _run_first_token_time(record: dict[str, Any]) -> float | None:
    """First token time of one packed run, or ``None`` when none qualifies.

    Member ``i``'s time is ``time0 + sum(dt[:i])`` (``dt`` holds the gaps
    between members). For a tool-call run with a ``name`` present, the
    first token is at ``time0`` immediately; otherwise the first
    non-empty fragment wins.
    """
    if record.get("type") == "tool-call-chunks" and "name" in record:
        return _record_time0(record)
    fragments = (
        record.get("args")
        if record.get("type") == "tool-call-chunks"
        else record.get("texts")
    )
    if not isinstance(fragments, list):
        return None
    time0 = _record_time0(record)
    if time0 is None:
        return None
    raw_dt = record.get("dt")
    gaps: list[float] = []
    if isinstance(raw_dt, list):
        gaps = [
            float(gap)
            if isinstance(gap, (int, float)) and not isinstance(gap, bool)
            else 0.0
            for gap in raw_dt
        ]
    current = time0
    for index, fragment in enumerate(fragments):
        if index > 0:
            current += gaps[index - 1] if index - 1 < len(gaps) else 0.0
        if isinstance(fragment, str) and fragment != "":
            return current
    return None


def first_token_time(stream: Any) -> float | None:
    """Time of the first token in one compact stream, or ``None``.

    Scans records in order and stops at the first qualifying member.
    """
    if not isinstance(stream, list):
        return None
    for record in stream:
        if not isinstance(record, dict):
            continue
        if record.get("type") == "chunk":
            time = record.get("time")
            if (
                _is_token_delta(record.get("chunk"))
                and isinstance(time, (int, float))
                and not isinstance(time, bool)
            ):
                return float(time)
            continue
        found = _run_first_token_time(record)
        if found is not None:
            return found
    return None


def _output_tokens(usage: Any) -> int | float | None:
    """Provider-reported output tokens, or ``None`` when unreported/invalid."""
    if not isinstance(usage, dict):
        return None
    value = usage.get("outputTokens")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value >= 0 else None


# ---------------------------------------------------------------------------
# Turn-outline preview.
# ---------------------------------------------------------------------------

PROMPT_PREVIEW_LIMIT = 50
RESPONSE_PREVIEW_LIMIT = 120


def preview(content: list[dict[str, Any]], limit: int) -> str:
    """Space-join text blocks, collapse whitespace, cap at ``limit``.

    Counts characters as code points. A result clipped by the budget
    ends in an ellipsis; the cut keeps the result within ``limit``
    characters.
    """
    text = ""
    unread = False
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        if len(text) >= limit * 2:
            unread = True
            break
        block_text = block.get("text")
        if not isinstance(block_text, str):
            continue
        clipped = len(block_text) > limit * 2
        chunk = block_text[: limit * 2] if clipped else block_text
        text += chunk if text == "" else f" {chunk}"
        if clipped:
            unread = True
            break
    normalized = _WHITESPACE_RUN.sub(" ", text).strip()
    if len(normalized) > limit - 1:
        return normalized[: limit - 1].rstrip() + "…"
    if unread:
        return normalized + "…"
    return normalized


# ---------------------------------------------------------------------------
# State / view schemas. Each validator raises ValueError on invalid input.
# ---------------------------------------------------------------------------

_TITLE_VALUE_ADAPTER = TypeAdapter(
    Annotated[str, StringConstraints(min_length=1)] | None,
    config=ConfigDict(strict=True),
)


def validate_title_value(value: Any) -> str | None:
    """A title string of length >= 1, or ``None``."""
    return _TITLE_VALUE_ADAPTER.validate_python(value)


def _require_absolute(value: str) -> str:
    if not os.path.isabs(value):
        raise ValueError("working directory must be an absolute path")
    return value


_WORKING_DIRECTORY_ADAPTER = TypeAdapter(
    Annotated[str, Field(min_length=1), AfterValidator(_require_absolute)] | None,
    config=ConfigDict(strict=True),
)


def validate_working_directory_value(value: Any) -> str | None:
    """An absolute path string of length >= 1, or ``None``."""
    return _WORKING_DIRECTORY_ADAPTER.validate_python(value)


class _TitleInputMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    seq: Annotated[int, Field(ge=0)]
    text: str


class _TitleInputState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    first: _TitleInputMessage | None
    count: Annotated[int, Field(ge=0)]
    last_seq: Annotated[int, Field(ge=0)] | None

    @model_validator(mode="after")
    def _check_invariant(self) -> Self:
        empty = self.first is None and self.last_seq is None and self.count == 0
        populated = (
            self.first is not None
            and self.last_seq is not None
            and self.count > 0
            and self.first.seq <= self.last_seq
        )
        if not empty and not populated:
            raise ValueError(
                "title input state must pair its count with first and last message seqs"
            )
        return self


_TITLE_INPUT_ADAPTER = TypeAdapter(_TitleInputState)


def validate_title_input_state(value: Any) -> dict[str, Any]:
    """Title-input aggregate plus its empty/populated cross-field invariant."""
    return _TITLE_INPUT_ADAPTER.validate_python(value).model_dump()


class _OpenStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    turn: Annotated[int, Field(ge=0)]
    step: Annotated[int, Field(ge=0)]
    start_time: Annotated[float, Field(ge=0)]
    first_token_time: Annotated[float, Field(ge=0)] | None


class _SessionStatsState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    turns: Annotated[int, Field(ge=0)]
    steps: Annotated[int, Field(ge=0)]
    llm_ms: Annotated[float, Field(ge=0)]
    tool_ms: Annotated[float, Field(ge=0)]
    ttft_ms: Annotated[float, Field(ge=0)]
    ttft_steps: Annotated[int, Field(ge=0)]
    decode_ms: Annotated[float, Field(ge=0)]
    decode_tokens: Annotated[int, Field(ge=0)]
    last_turn: Annotated[int, Field(ge=0)] | None
    open_step: _OpenStep | None
    pending_calls: dict[str, Annotated[float, Field(ge=0)]]


class _SessionStatsView(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    turns: Annotated[int, Field(ge=0)]
    steps: Annotated[int, Field(ge=0)]
    llm_ms: Annotated[float, Field(ge=0)]
    tool_ms: Annotated[float, Field(ge=0)]
    ttft_ms: Annotated[float, Field(ge=0)]
    ttft_steps: Annotated[int, Field(ge=0)]
    decode_ms: Annotated[float, Field(ge=0)]
    decode_tokens: Annotated[int, Field(ge=0)]


_SESSION_STATS_STATE_ADAPTER = TypeAdapter(_SessionStatsState)
_SESSION_STATS_VIEW_ADAPTER = TypeAdapter(_SessionStatsView)


def validate_session_stats_state(value: Any) -> dict[str, Any]:
    """Session-stats fold state: the 8 totals plus in-flight boundaries."""
    return _SESSION_STATS_STATE_ADAPTER.validate_python(value).model_dump()


def validate_session_stats_view(value: Any) -> dict[str, Any]:
    """Session-stats wire view: exactly the 8 totals."""
    return _SESSION_STATS_VIEW_ADAPTER.validate_python(value).model_dump()


class _TurnOutlineEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    turn: Annotated[int, Field(ge=0)]
    seq: Annotated[int, Field(ge=0)]
    prompt: Annotated[str, Field(max_length=PROMPT_PREVIEW_LIMIT)]
    response: Annotated[str, Field(max_length=RESPONSE_PREVIEW_LIMIT)]


class _TurnOutlineState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    turns: list[_TurnOutlineEntry]
    draft: Annotated[str, Field(max_length=RESPONSE_PREVIEW_LIMIT)]


_TURN_OUTLINE_STATE_ADAPTER = TypeAdapter(_TurnOutlineState)
_TURN_OUTLINE_VIEW_ADAPTER = TypeAdapter(list[_TurnOutlineEntry])


def _check_turn_order(turns: list[dict[str, Any]]) -> None:
    """Turn-outline entries must be strictly increasing by turn."""
    previous = -1
    for entry in turns:
        if entry["turn"] <= previous:
            raise ValueError("turn outline entries must be strictly increasing by turn")
        previous = entry["turn"]


def validate_turn_outline_state(value: Any) -> dict[str, Any]:
    """Turn-outline fold state: entries plus the uncommitted draft."""
    state = _TURN_OUTLINE_STATE_ADAPTER.validate_python(value).model_dump()
    _check_turn_order(state["turns"])
    return state


def validate_turn_outline_view(value: Any) -> list[dict[str, Any]]:
    """Turn-outline wire view: the entries array only (no draft)."""
    entries = _TURN_OUTLINE_VIEW_ADAPTER.validate_python(value)
    dumped = [entry.model_dump() for entry in entries]
    _check_turn_order(dumped)
    return dumped


# ---------------------------------------------------------------------------
# title — latest title text or None (state is the wire value).
# ---------------------------------------------------------------------------


def _init_title(header: dict[str, Any], inherited_event_count: int) -> str | None:
    return None


def _apply_title(state: str | None, event: SessionEvent) -> str | None:
    if event.type != "session/title":
        return state
    title = event.payload.get("title")
    # An empty title is kept as state (it fails the schema on read
    # paths); only a non-string title is not this unit's event.
    return title if isinstance(title, str) else state


def _title_view(state: str | None) -> str | None:
    return state


TITLE_PROJECTION: ProjectionDefinition[str | None] = ProjectionDefinition(
    key="title",
    state_schema=validate_title_value,
    init=_init_title,
    apply=_apply_title,
    wire=WireView(view_schema=validate_title_value, view=_title_view),
    state_version=1,
)


# ---------------------------------------------------------------------------
# titleInput — eligible-message aggregate (host-only).
# ---------------------------------------------------------------------------


def _init_title_input(
    header: dict[str, Any], inherited_event_count: int
) -> dict[str, Any]:
    return {"first": None, "count": 0, "last_seq": None}


def _apply_title_input(state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
    message = _eligible_title_message(event)
    if message is None:
        return state
    return {
        "first": state["first"] if state["first"] is not None else message,
        "count": state["count"] + 1,
        "last_seq": message["seq"],
    }


TITLE_INPUT_PROJECTION: ProjectionDefinition[dict[str, Any]] = ProjectionDefinition(
    key="titleInput",
    state_schema=validate_title_input_state,
    init=_init_title_input,
    apply=_apply_title_input,
    state_version=3,
)


# ---------------------------------------------------------------------------
# sessionStats — whole-log counts and wall times.
# ---------------------------------------------------------------------------


def _init_session_stats(
    header: dict[str, Any], inherited_event_count: int
) -> dict[str, Any]:
    return {
        "turns": 0,
        "steps": 0,
        "llm_ms": 0.0,
        "tool_ms": 0.0,
        "ttft_ms": 0.0,
        "ttft_steps": 0,
        "decode_ms": 0.0,
        "decode_tokens": 0,
        "last_turn": None,
        "open_step": None,
        "pending_calls": {},
    }


def _turn_step(payload: dict[str, Any]) -> tuple[int, int] | None:
    """One event's (turn, step) pair, or ``None`` when it carries none."""
    turn = payload.get("turn")
    step = payload.get("step")
    if isinstance(turn, bool) or not isinstance(turn, int):
        return None
    if isinstance(step, bool) or not isinstance(step, int):
        return None
    return (turn, step)


def _apply_session_stats(state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
    payload = event.payload
    event_time = event.timestamp * 1000
    if event.type == "step/start":
        turn_step = _turn_step(payload)
        if turn_step is None:
            return state
        turn, step = turn_step
        # Replaces any open step, no guard: a step/start arriving
        # while a step is open discards the earlier boundary.
        return {
            **state,
            "open_step": {
                "turn": turn,
                "step": step,
                "start_time": event_time,
                "first_token_time": None,
            },
        }
    if event.type == "assistant/attempt":
        open_step = state["open_step"]
        turn_step = _turn_step(payload)
        if open_step is None or turn_step is None:
            return state
        if open_step["turn"] != turn_step[0] or open_step["step"] != turn_step[1]:
            return state
        first = first_token_time(payload.get("stream"))
        if open_step["first_token_time"] is not None or first is None:
            return state
        return {**state, "open_step": {**open_step, "first_token_time": first}}
    if event.type == "assistant/message":
        open_step = state["open_step"]
        turn_step = _turn_step(payload)
        if open_step is None or turn_step is None:
            return state
        if open_step["turn"] != turn_step[0] or open_step["step"] != turn_step[1]:
            return state
        first_token = open_step["first_token_time"]
        if first_token is None:
            first_token = first_token_time(payload.get("stream"))
        # One assembled message per step: closing the boundary means a
        # defensive duplicate cannot accrue twice.
        next_state: dict[str, Any] = {
            **state,
            "llm_ms": state["llm_ms"] + max(0.0, event_time - open_step["start_time"]),
            "open_step": None,
        }
        if first_token is not None:
            next_state["ttft_ms"] = state["ttft_ms"] + max(
                0.0, first_token - open_step["start_time"]
            )
            next_state["ttft_steps"] = state["ttft_steps"] + 1
            tokens = _output_tokens(payload.get("usage"))
            if tokens is not None:
                next_state["decode_ms"] = state["decode_ms"] + max(
                    0.0, event_time - first_token
                )
                next_state["decode_tokens"] = state["decode_tokens"] + tokens
        return next_state
    if event.type == "tool/call":
        call_id = payload.get("callId")
        if not isinstance(call_id, str):
            return state
        return {
            **state,
            "pending_calls": {**state["pending_calls"], call_id: event_time},
        }
    if event.type == "tool/result":
        message = payload.get("message")
        source = message.get("source") if isinstance(message, dict) else None
        call_id = source.get("callId") if isinstance(source, dict) else None
        if not isinstance(call_id, str) or call_id not in state["pending_calls"]:
            return state
        dispatched = state["pending_calls"][call_id]
        pending = {
            key: value
            for key, value in state["pending_calls"].items()
            if key != call_id
        }
        return {
            **state,
            "tool_ms": state["tool_ms"] + max(0.0, event_time - dispatched),
            "pending_calls": pending,
        }
    if event.type == "step/end":
        end_turn = payload.get("turn")
        if isinstance(end_turn, bool) or not isinstance(end_turn, int):
            return state
        # No open-step guard: every step/end counts, including cancelled
        # steps (which accrue no time — they assemble no message).
        return {
            **state,
            "turns": state["turns"] + (0 if state["last_turn"] == end_turn else 1),
            "steps": state["steps"] + 1,
            "last_turn": end_turn,
            "open_step": None,
        }
    if event.type == "turn/end":
        if not state["pending_calls"]:
            return state
        # Results always land within their turn; leftovers belong to a
        # cancelled or failed turn.
        return {**state, "pending_calls": {}}
    return state


def _session_stats_view(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "turns": state["turns"],
        "steps": state["steps"],
        "llm_ms": state["llm_ms"],
        "tool_ms": state["tool_ms"],
        "ttft_ms": state["ttft_ms"],
        "ttft_steps": state["ttft_steps"],
        "decode_ms": state["decode_ms"],
        "decode_tokens": state["decode_tokens"],
    }


SESSION_STATS_PROJECTION: ProjectionDefinition[dict[str, Any]] = ProjectionDefinition(
    key="sessionStats",
    state_schema=validate_session_stats_state,
    init=_init_session_stats,
    apply=_apply_session_stats,
    wire=WireView(
        view_schema=validate_session_stats_view,
        view=_session_stats_view,
        summary=False,
    ),
    state_version=1,
)


# ---------------------------------------------------------------------------
# turnOutline — per-turn prompt/response previews.
# ---------------------------------------------------------------------------


def _init_turn_outline(
    header: dict[str, Any], inherited_event_count: int
) -> dict[str, Any]:
    return {"turns": [], "draft": ""}


def _apply_turn_outline(state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
    payload = event.payload
    if event.type == "turn/start":
        turn = payload.get("turn")
        if isinstance(turn, bool) or not isinstance(turn, int):
            return state
        turns = state["turns"]
        last = turns[-1] if turns else None
        # Order guard: a boundary that does not advance the turn number
        # keeps the outline sorted; a retried turn's previews land on
        # the standing entry.
        if last is not None and turn <= last["turn"]:
            return state
        return {
            "turns": [
                *turns,
                {"turn": turn, "seq": event.seq, "prompt": "", "response": ""},
            ],
            "draft": "",
        }
    if event.type == "user/message":
        source = payload.get("source")
        if not isinstance(source, dict) or source.get("kind") != "user":
            return state
        turns = state["turns"]
        last = turns[-1] if turns else None
        # Only the newest turn can still be waiting for its opening
        # prompt; later human messages in the same turn (steering)
        # keep the first preview.
        if last is None or last["prompt"] != "":
            return state
        content = payload.get("content")
        if not isinstance(content, list):
            return state
        prompt = preview(content, PROMPT_PREVIEW_LIMIT)
        if prompt == "":
            return state
        return {
            "turns": [*turns[:-1], {**last, "prompt": prompt}],
            "draft": state["draft"],
        }
    if event.type == "assistant/message":
        message = payload.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return state
        draft = preview(content, RESPONSE_PREVIEW_LIMIT)
        if draft == "" or draft == state["draft"]:
            return state
        # Draft-only applies keep the turns list identity, so the
        # identity-gated change feed stays quiet between boundaries.
        return {"turns": state["turns"], "draft": draft}
    if event.type == "turn/end":
        if state["draft"] == "":
            return state
        turns = state["turns"]
        last = turns[-1] if turns else None
        if last is None or last["response"] == state["draft"]:
            return {"turns": turns, "draft": ""}
        return {
            "turns": [*turns[:-1], {**last, "response": state["draft"]}],
            "draft": "",
        }
    return state


def _turn_outline_view(state: dict[str, Any]) -> Any:
    return state["turns"]


TURN_OUTLINE_PROJECTION: ProjectionDefinition[dict[str, Any]] = ProjectionDefinition(
    key="turnOutline",
    state_schema=validate_turn_outline_state,
    init=_init_turn_outline,
    apply=_apply_turn_outline,
    wire=WireView(
        view_schema=validate_turn_outline_view,
        view=_turn_outline_view,
        summary=False,
    ),
    state_version=2,
)


# ---------------------------------------------------------------------------
# workingDirectory — current effective directory.
# ---------------------------------------------------------------------------


def _init_working_directory(
    header: dict[str, Any], inherited_event_count: int
) -> str | None:
    cwd = header.get("cwd")
    return cwd if isinstance(cwd, str) and cwd else None


def _apply_working_directory(state: str | None, event: SessionEvent) -> str | None:
    if event.type != "working-directory/change":
        return state
    cwd = event.payload.get("cwd")
    return cwd if isinstance(cwd, str) else state


def _working_directory_view(state: str | None) -> str | None:
    return state


WORKING_DIRECTORY_PROJECTION: ProjectionDefinition[str | None] = ProjectionDefinition(
    key="workingDirectory",
    state_schema=validate_working_directory_value,
    init=_init_working_directory,
    apply=_apply_working_directory,
    wire=WireView(
        view_schema=validate_working_directory_value,
        view=_working_directory_view,
    ),
    state_version=1,
)
