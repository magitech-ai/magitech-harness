"""Unit tests for the five projection units."""

from typing import Any

import pytest
from anima import Context

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.projection_registry import (
    ProjectionDefinition,
    SessionProjectionRegistry,
)
from magitech_harness.session.projections import (
    SESSION_STATS_PROJECTION,
    TITLE_INPUT_PROJECTION,
    TITLE_PROJECTION,
    TURN_OUTLINE_PROJECTION,
    WORKING_DIRECTORY_PROJECTION,
    _output_tokens,
    first_token_time,
    preview,
    validate_session_stats_state,
    validate_title_input_state,
    validate_title_value,
    validate_turn_outline_state,
    validate_turn_outline_view,
    validate_working_directory_value,
)
from magitech_harness.session.store import Session

HEADER: dict[str, Any] = {
    "version": 4,
    "id": "s1",
    "createdAt": 1700000000000,
    "cwd": "/home/malcom/work",
    "isSeeded": False,
    "delegationDepth": 0,
}

ALL_DEFINITIONS = (
    TITLE_PROJECTION,
    TITLE_INPUT_PROJECTION,
    SESSION_STATS_PROJECTION,
    TURN_OUTLINE_PROJECTION,
    WORKING_DIRECTORY_PROJECTION,
)


def _event(
    event_type: str,
    payload: dict[str, Any],
    seq: int,
    ms: int,
    *,
    surface_op: str | None = None,
) -> SessionEvent:
    return SessionEvent(
        id=f"s1:{seq}",
        session_id="s1",
        type=event_type,
        timestamp=ms / 1000,
        payload=payload,
        seq=seq,
        surface_op=surface_op,
    )


def _user_message(text: str, seq: int, ms: int) -> SessionEvent:
    return _event(
        "user/message",
        {
            "content": [{"type": "text", "text": text}],
            "source": {"kind": "user"},
        },
        seq,
        ms,
        surface_op="append",
    )


def _assistant_message(
    text: str,
    seq: int,
    ms: int,
    *,
    turn: int = 1,
    step: int = 1,
    stream: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
) -> SessionEvent:
    payload: dict[str, Any] = {
        "turn": turn,
        "step": step,
        "message": {
            "id": f"a{seq}",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    }
    if stream is not None:
        payload["stream"] = stream
    if usage is not None:
        payload["usage"] = usage
    return _event("assistant/message", payload, seq, ms, surface_op="append")


def _golden_events() -> list[SessionEvent]:
    """The spec §5.1 Scenario A log (10 events, exact timestamps)."""
    return [
        _event("turn/start", {"turn": 1}, 0, 1000),
        _user_message("Hello world", 1, 1100),
        _event("step/start", {"turn": 1, "step": 1}, 2, 1200),
        _assistant_message(
            "Hi there",
            3,
            1500,
            stream=[
                {
                    "type": "chunk",
                    "time": 1300,
                    "chunk": {"type": "text-delta", "index": 0, "text": "Hi"},
                }
            ],
            usage={"outputTokens": 5},
        ),
        _event(
            "tool/call",
            {"turn": 1, "step": 1, "callId": "c1", "name": "add", "arguments": {}},
            4,
            1600,
        ),
        _event(
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "m1",
                    "role": "tool",
                    "source": {"kind": "tool", "callId": "c1"},
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "c1",
                            "content": [{"type": "text", "text": "3"}],
                        }
                    ],
                },
            },
            5,
            1800,
            surface_op="append",
        ),
        _event("step/end", {"turn": 1, "step": 1}, 6, 1900),
        _event("turn/end", {"turn": 1, "reason": {"kind": "completed"}}, 7, 2000),
        _event(
            "session/title",
            {
                "title": "Hello world",
                "messageSeqs": [1],
                "source": {"kind": "fallback"},
            },
            8,
            2100,
        ),
        _event("working-directory/change", {"cwd": "/tmp/proj"}, 9, 2200),
    ]


def _fold(
    definition: ProjectionDefinition[Any],
    events: list[SessionEvent],
    header: dict[str, Any] | None = None,
) -> Any:
    state = definition.init(header if header is not None else HEADER, 0)
    for event in events:
        state = definition.apply(state, event)
    return state


def _registry() -> tuple[Context, SessionProjectionRegistry]:
    ctx = Context()
    registry = SessionProjectionRegistry(ctx)
    for definition in ALL_DEFINITIONS:
        registry.register(definition)
    return ctx, registry


def _expected_stats_view() -> dict[str, Any]:
    return {
        "turns": 1,
        "steps": 1,
        "llm_ms": 300.0,
        "tool_ms": 200.0,
        "ttft_ms": 100.0,
        "ttft_steps": 1,
        "decode_ms": 200.0,
        "decode_tokens": 5,
    }


# ---------------------------------------------------------------------------
# Scenario A — the five-unit golden log.
# ---------------------------------------------------------------------------


def test_golden_log_pure_folds() -> None:
    """Each unit's fold reproduces the hand-computed Scenario A values."""
    events = _golden_events()
    assert _fold(TITLE_PROJECTION, events) == "Hello world"
    assert _fold(TITLE_INPUT_PROJECTION, events) == {
        "first": {"seq": 1, "text": "Hello world"},
        "count": 1,
        "last_seq": 1,
    }
    assert (
        SESSION_STATS_PROJECTION.wire is not None
        and SESSION_STATS_PROJECTION.wire.view(_fold(SESSION_STATS_PROJECTION, events))
        == _expected_stats_view()
    )
    assert _fold(TURN_OUTLINE_PROJECTION, events) == {
        "turns": [
            {"turn": 1, "seq": 0, "prompt": "Hello world", "response": "Hi there"}
        ],
        "draft": "",
    }
    assert _fold(WORKING_DIRECTORY_PROJECTION, events) == "/tmp/proj"


def test_golden_log_restore_agrees_with_pure_fold() -> None:
    """Cold restore over the golden log yields the same snapshot."""
    _, registry = _registry()
    events = _golden_events()
    restored = registry.restore({}, events, 0, HEADER, 0)
    assert restored.snapshot == {
        "as_of_seq": 9,
        "values": {
            "title": "Hello world",
            "sessionStats": _expected_stats_view(),
            "turnOutline": [
                {
                    "turn": 1,
                    "seq": 0,
                    "prompt": "Hello world",
                    "response": "Hi there",
                }
            ],
            "workingDirectory": "/tmp/proj",
        },
    }
    assert restored.checkpoint["title"] == {"ver": 1, "seq": 9, "val": "Hello world"}
    assert restored.checkpoint["titleInput"]["ver"] == 3
    assert restored.checkpoint["sessionStats"]["seq"] == 9
    assert registry.summary_keys() == frozenset({"title", "workingDirectory"})


def test_golden_log_live_drive() -> None:
    """The eager drive over exact-timestamp events matches the cold fold."""
    ctx, registry = _registry()
    events = _golden_events()
    session = Session.create("s1", seed=list(events), header=dict(HEADER))
    # The constructor appends one end-seed marker (seq 10); drive all of it.
    for event in session.events:
        ctx.emit("session/event", session, event)
    snapshot = registry.snapshot(session)
    assert snapshot["as_of_seq"] == 10
    assert snapshot["values"] == {
        "title": "Hello world",
        "sessionStats": _expected_stats_view(),
        "turnOutline": [
            {"turn": 1, "seq": 0, "prompt": "Hello world", "response": "Hi there"}
        ],
        "workingDirectory": "/tmp/proj",
    }
    assert registry.cached_snapshot(session) == snapshot


# ---------------------------------------------------------------------------
# title
# ---------------------------------------------------------------------------


def test_title_latest_wins_and_empty_title_is_kept_as_state() -> None:
    """Latest session/title wins; an empty title stays empty in the fold."""
    state = _fold(
        TITLE_PROJECTION,
        [
            _event(
                "session/title",
                {"title": "First", "messageSeqs": [], "source": {"kind": "user"}},
                0,
                1000,
            ),
            _event(
                "session/title",
                {"title": "Second", "messageSeqs": [], "source": {"kind": "user"}},
                1,
                2000,
            ),
        ],
    )
    assert state == "Second"
    empty = _fold(
        TITLE_PROJECTION,
        [
            _event(
                "session/title",
                {"title": "", "messageSeqs": [], "source": {"kind": "user"}},
                0,
                1000,
            )
        ],
    )
    assert empty == ""


def test_title_ignores_other_events_and_non_string_titles() -> None:
    """Any other event — or a title event without a string — is a no-op."""
    state = TITLE_PROJECTION.init(HEADER, 0)
    assert (
        TITLE_PROJECTION.apply(state, _event("turn/start", {"turn": 1}, 0, 1000))
        is state
    )
    assert (
        TITLE_PROJECTION.apply(state, _event("session/title", {"title": 7}, 0, 1000))
        is state
    )


def test_title_value_schema() -> None:
    """The title schema takes a non-empty string or None, nothing else."""
    assert validate_title_value("x") == "x"
    assert validate_title_value(None) is None
    for bad in ("", 5, True, [], {}):
        with pytest.raises(ValueError):
            validate_title_value(bad)


# ---------------------------------------------------------------------------
# titleInput
# ---------------------------------------------------------------------------


def test_title_input_eligibility() -> None:
    """Only human prompts with cleanable text count; first is never replaced."""
    events = [
        _event(
            "user/message",
            {
                "content": [{"type": "text", "text": "injected"}],
                "source": {"kind": "plugin:compact"},
            },
            0,
            1000,
            surface_op="append",
        ),
        _event(
            "user/message",
            {
                "content": [{"type": "text", "text": "   \n\t  "}],
                "source": {"kind": "user"},
            },
            1,
            1100,
            surface_op="append",
        ),
        _event(
            "user/message",
            {
                "content": [{"type": "text", "text": "\x1b[31m\x1b[0m"}],
                "source": {"kind": "user"},
            },
            2,
            1200,
            surface_op="append",
        ),
        _user_message("first prompt", 3, 1300),
        _user_message("second prompt", 4, 1400),
        _event("turn/start", {"turn": 1}, 5, 1500),
    ]
    assert _fold(TITLE_INPUT_PROJECTION, events) == {
        "first": {"seq": 3, "text": "first prompt"},
        "count": 2,
        "last_seq": 4,
    }


def test_title_input_keeps_raw_text_and_cleans_only_for_emptiness() -> None:
    """The stored text is raw joined text; cleaning decides eligibility only."""
    events = [_user_message("  hello   world  ", 0, 1000)]
    assert _fold(TITLE_INPUT_PROJECTION, events) == {
        "first": {"seq": 0, "text": "  hello   world  "},
        "count": 1,
        "last_seq": 0,
    }
    # Escape-coloured text is eligible and stored raw.
    coloured = _event(
        "user/message",
        {
            "content": [{"type": "text", "text": "\x1b[31mred\x1b[0m"}],
            "source": {"kind": "user"},
        },
        0,
        1000,
        surface_op="append",
    )
    assert _fold(TITLE_INPUT_PROJECTION, [coloured])["first"]["text"] == (
        "\x1b[31mred\x1b[0m"
    )


def test_title_input_ignores_malformed_user_messages() -> None:
    """A user/message without content or source is not this unit's event."""
    state = TITLE_INPUT_PROJECTION.init(HEADER, 0)
    assert (
        TITLE_INPUT_PROJECTION.apply(
            state, _event("user/message", {"text": "hi"}, 0, 1000)
        )
        is state
    )
    assert (
        TITLE_INPUT_PROJECTION.apply(
            state,
            _event(
                "user/message",
                {"content": [{"type": "text", "text": "hi"}]},
                0,
                1000,
            ),
        )
        is state
    )
    # A human message whose content is not a list is not eligible either.
    assert (
        TITLE_INPUT_PROJECTION.apply(
            state,
            _event(
                "user/message",
                {"content": "hi", "source": {"kind": "user"}},
                0,
                1000,
            ),
        )
        is state
    )
    assert TITLE_INPUT_PROJECTION.state_version == 3
    assert TITLE_INPUT_PROJECTION.wire is None


def test_title_input_state_schema() -> None:
    """The aggregate schema enforces the empty/populated invariant."""
    assert validate_title_input_state(
        {"first": None, "count": 0, "last_seq": None}
    ) == {
        "first": None,
        "count": 0,
        "last_seq": None,
    }
    assert (
        validate_title_input_state(
            {"first": {"seq": 1, "text": "hi"}, "count": 2, "last_seq": 3}
        )["count"]
        == 2
    )
    for bad in (
        {"first": None, "count": 1, "last_seq": None},
        {"first": {"seq": 5, "text": "hi"}, "count": 1, "last_seq": 3},
        {"first": {"seq": 1, "text": "hi"}, "count": 0, "last_seq": 1},
        {"first": None, "count": 0, "last_seq": 1},
        {"first": {"seq": -1, "text": "hi"}, "count": 1, "last_seq": 1},
        {"first": {"seq": 1, "text": "hi"}, "count": 1, "last_seq": 1, "extra": 1},
    ):
        with pytest.raises(ValueError):
            validate_title_input_state(bad)


# ---------------------------------------------------------------------------
# sessionStats
# ---------------------------------------------------------------------------


def _stats_view(events: list[SessionEvent]) -> dict[str, Any]:
    state = _fold(SESSION_STATS_PROJECTION, events)
    assert SESSION_STATS_PROJECTION.wire is not None
    return SESSION_STATS_PROJECTION.wire.view(state)


def test_session_stats_malformed_step_boundary_is_ignored() -> None:
    """Step boundaries without an integer (turn, step) pair fold to nothing."""
    events = [
        _event("step/start", {"turn": 1, "step": "x"}, 0, 1200),
        _event("step/start", {"turn": "x", "step": 1}, 1, 1300),
        _event("assistant/attempt", {"turn": 1, "step": 1, "stream": []}, 2, 1400),
    ]
    assert _stats_view(events) == {
        "turns": 0,
        "steps": 0,
        "llm_ms": 0.0,
        "tool_ms": 0.0,
        "ttft_ms": 0.0,
        "ttft_steps": 0,
        "decode_ms": 0.0,
        "decode_tokens": 0,
    }


def test_session_stats_first_token_without_usage_accrues_no_decode() -> None:
    """A first token with no usage counts ttft but no decode time/tokens."""
    events = [
        _event("step/start", {"turn": 1, "step": 1}, 0, 1200),
        _assistant_message(
            "Hi",
            1,
            1500,
            stream=[
                {
                    "type": "chunk",
                    "time": 1300,
                    "chunk": {"type": "text-delta", "index": 0, "text": "Hi"},
                }
            ],
        ),
        _event("step/end", {"turn": 1, "step": 1}, 2, 1600),
    ]
    assert _stats_view(events) == {
        "turns": 1,
        "steps": 1,
        "llm_ms": 300.0,
        "tool_ms": 0.0,
        "ttft_ms": 100.0,
        "ttft_steps": 1,
        "decode_ms": 0.0,
        "decode_tokens": 0,
    }


def test_session_stats_scenario_b_cancelled_step_and_ghost_result() -> None:
    """A cancelled step accrues no time; an unmatched result is a no-op."""
    events = [
        _event("turn/start", {"turn": 1}, 0, 1000),
        _event("step/start", {"turn": 1, "step": 1}, 1, 1200),
        _event(
            "assistant/attempt",
            {
                "turn": 1,
                "step": 1,
                "stream": [
                    {
                        "type": "chunk",
                        "time": 1300,
                        "chunk": {"type": "text-delta", "index": 0, "text": "…"},
                    }
                ],
            },
            2,
            1400,
        ),
        _event("step/end", {"turn": 1, "step": 1}, 3, 1900),
        _event("turn/end", {"turn": 1}, 4, 2000),
        _event(
            "tool/result",
            {
                "turn": 1,
                "step": 1,
                "message": {
                    "id": "m9",
                    "role": "tool",
                    "source": {"kind": "tool", "callId": "ghost"},
                    "content": [],
                },
            },
            5,
            2100,
            surface_op="append",
        ),
    ]
    assert _stats_view(events) == {
        "turns": 1,
        "steps": 1,
        "llm_ms": 0.0,
        "tool_ms": 0.0,
        "ttft_ms": 0.0,
        "ttft_steps": 0,
        "decode_ms": 0.0,
        "decode_tokens": 0,
    }


def test_session_stats_first_token_survives_retry() -> None:
    """An attempt's first token is preferred over the message's own stream."""
    events = [
        _event("step/start", {"turn": 1, "step": 1}, 0, 1200),
        _event(
            "assistant/attempt",
            {
                "turn": 1,
                "step": 1,
                "stream": [
                    {
                        "type": "chunk",
                        "time": 1300,
                        "chunk": {"type": "text-delta", "index": 0, "text": "x"},
                    }
                ],
            },
            1,
            1350,
        ),
        _assistant_message(
            "retry text",
            2,
            1500,
            stream=[
                {
                    "type": "chunk",
                    "time": 1400,
                    "chunk": {"type": "text-delta", "index": 0, "text": "y"},
                }
            ],
            usage={"outputTokens": 3},
        ),
        _event("step/end", {"turn": 1, "step": 1}, 3, 1600),
    ]
    assert _stats_view(events) == {
        "turns": 1,
        "steps": 1,
        "llm_ms": 300.0,
        "tool_ms": 0.0,
        "ttft_ms": 100.0,
        "ttft_steps": 1,
        "decode_ms": 200.0,
        "decode_tokens": 3,
    }


def test_session_stats_duplicate_message_accrues_once() -> None:
    """A defensive duplicate message for the same step cannot accrue twice."""
    message = _assistant_message(
        "Hi",
        1,
        1500,
        stream=[
            {
                "type": "chunk",
                "time": 1300,
                "chunk": {"type": "text-delta", "index": 0, "text": "Hi"},
            }
        ],
        usage={"outputTokens": 5},
    )
    events = [
        _event("step/start", {"turn": 1, "step": 1}, 0, 1200),
        message,
        _event(
            "assistant/message",
            message.payload,
            2,
            1600,
            surface_op="append",
        ),
        _event("step/end", {"turn": 1, "step": 1}, 3, 1700),
    ]
    view = _stats_view(events)
    assert view["llm_ms"] == 300.0
    assert view["decode_tokens"] == 5


def test_session_stats_step_start_overwrites_open_step() -> None:
    """A step/start arriving while a step is open replaces the boundary."""
    events = [
        _event("step/start", {"turn": 1, "step": 1}, 0, 1000),
        _event("step/start", {"turn": 1, "step": 2}, 1, 1500),
        _assistant_message("done", 2, 1800, turn=1, step=2),
        _event("step/end", {"turn": 1, "step": 2}, 3, 1900),
    ]
    view = _stats_view(events)
    assert view["llm_ms"] == 300.0
    assert view["steps"] == 1


def test_session_stats_tool_call_overwrite_and_turn_end_clears() -> None:
    """A duplicate callId overwrites (last dispatch wins); turn/end clears."""
    events = [
        _event("tool/call", {"callId": "c1", "name": "add"}, 0, 1000),
        _event("tool/call", {"callId": "c1", "name": "add"}, 1, 1500),
        _event(
            "tool/result",
            {
                "message": {
                    "id": "m1",
                    "role": "tool",
                    "source": {"kind": "tool", "callId": "c1"},
                    "content": [],
                }
            },
            2,
            1700,
            surface_op="append",
        ),
    ]
    assert _stats_view(events)["tool_ms"] == 200.0
    # turn/end with a pending call drops it without accruing.
    pending = [
        _event("tool/call", {"callId": "c9", "name": "x"}, 0, 1000),
        _event("turn/end", {"turn": 1}, 1, 2000),
    ]
    state = _fold(SESSION_STATS_PROJECTION, pending)
    assert state["pending_calls"] == {}
    # turn/end with nothing pending returns the same reference.
    state2 = _fold(SESSION_STATS_PROJECTION, [_event("turn/end", {"turn": 1}, 0, 1000)])
    assert state2["pending_calls"] == {}


def test_session_stats_turns_count_distinct_turns() -> None:
    """Steps count at step/end; turns count distinct turns with a closed step."""
    events = [
        _event("step/end", {"turn": 1, "step": 1}, 0, 1000),
        _event("step/end", {"turn": 1, "step": 2}, 1, 1100),
        _event("step/end", {"turn": 2, "step": 1}, 2, 1200),
    ]
    view = _stats_view(events)
    assert view["turns"] == 2
    assert view["steps"] == 3


def test_session_stats_ignores_mismatched_and_malformed_events() -> None:
    """Mismatched turn/step, missing fields, and ghost results are no-ops."""
    state = SESSION_STATS_PROJECTION.init(HEADER, 0)
    opened = SESSION_STATS_PROJECTION.apply(
        state, _event("step/start", {"turn": 1, "step": 1}, 0, 1000)
    )
    assert opened is not state
    # attempt for another step, message for another turn, malformed starts.
    assert (
        SESSION_STATS_PROJECTION.apply(
            opened,
            _event("assistant/attempt", {"turn": 1, "step": 2, "stream": []}, 1, 1100),
        )
        is opened
    )
    assert (
        SESSION_STATS_PROJECTION.apply(
            opened, _assistant_message("x", 1, 1100, turn=2, step=1)
        )
        is opened
    )
    assert (
        SESSION_STATS_PROJECTION.apply(
            state, _event("step/start", {"turn": 1}, 0, 1000)
        )
        is state
    )
    assert (
        SESSION_STATS_PROJECTION.apply(
            state, _event("step/end", {"turn": "one"}, 0, 1000)
        )
        is state
    )
    assert (
        SESSION_STATS_PROJECTION.apply(
            state, _event("tool/call", {"name": "x"}, 0, 1000)
        )
        is state
    )
    assert (
        SESSION_STATS_PROJECTION.apply(
            state,
            _event(
                "tool/result",
                {"message": {"source": {}}},
                0,
                1000,
                surface_op="append",
            ),
        )
        is state
    )
    assert (
        SESSION_STATS_PROJECTION.apply(
            state, _event("tool/result", {}, 0, 1000, surface_op="append")
        )
        is state
    )
    # An attempt that records no token, and one after a token is already set.
    no_token = SESSION_STATS_PROJECTION.apply(
        opened,
        _event("assistant/attempt", {"turn": 1, "step": 1, "stream": []}, 1, 1100),
    )
    assert no_token is opened
    with_token = SESSION_STATS_PROJECTION.apply(
        opened,
        _event(
            "assistant/attempt",
            {
                "turn": 1,
                "step": 1,
                "stream": [
                    {
                        "type": "chunk",
                        "time": 1050,
                        "chunk": {"type": "text-delta", "index": 0, "text": "x"},
                    }
                ],
            },
            1,
            1100,
        ),
    )
    assert with_token is not opened
    assert (
        SESSION_STATS_PROJECTION.apply(
            with_token,
            _event(
                "assistant/attempt",
                {
                    "turn": 1,
                    "step": 1,
                    "stream": [
                        {
                            "type": "chunk",
                            "time": 1060,
                            "chunk": {"type": "text-delta", "index": 0, "text": "y"},
                        }
                    ],
                },
                2,
                1200,
            ),
        )
        is with_token
    )


def test_first_token_time_records() -> None:
    """Chunk and packed-run records resolve to the first qualifying time."""
    assert first_token_time(None) is None
    assert first_token_time([]) is None
    assert (
        first_token_time(
            [
                {
                    "type": "chunk",
                    "time": 100,
                    "chunk": {"type": "block-start", "blockType": "text"},
                }
            ]
        )
        is None
    )
    assert (
        first_token_time(
            [
                {
                    "type": "chunk",
                    "time": 100,
                    "chunk": {"type": "text-delta", "index": 0, "text": ""},
                }
            ]
        )
        is None
    )
    assert (
        first_token_time(
            [
                {
                    "type": "chunk",
                    "time": 100,
                    "chunk": {"type": "reasoning-delta", "index": 0, "text": "r"},
                }
            ]
        )
        == 100.0
    )
    assert (
        first_token_time(
            [
                {
                    "type": "chunk",
                    "time": 100,
                    "chunk": {
                        "type": "tool-call-delta",
                        "index": 0,
                        "argumentsDelta": "",
                    },
                }
            ]
        )
        is None
    )
    assert (
        first_token_time(
            [
                {
                    "type": "chunk",
                    "time": 100,
                    "chunk": {
                        "type": "tool-call-delta",
                        "index": 0,
                        "name": "add",
                        "argumentsDelta": "",
                    },
                }
            ]
        )
        == 100.0
    )
    assert (
        first_token_time(
            [
                {
                    "type": "chunk",
                    "time": "x",
                    "chunk": {"type": "text-delta", "index": 0, "text": "t"},
                }
            ]
        )
        is None
    )
    assert first_token_time([{"type": "chunk", "time": 100, "chunk": None}]) is None
    assert first_token_time([{"type": "chunk", "time": 100, "chunk": "text"}]) is None
    assert first_token_time(["not-a-record"]) is None
    # Packed text run: member times are time0 + cumulative dt gaps.
    assert (
        first_token_time(
            [
                {
                    "type": "text-chunks",
                    "time0": 1000,
                    "dt": [50, 25],
                    "texts": ["", "a", "b"],
                    "index": 0,
                }
            ]
        )
        == 1050.0
    )
    assert (
        first_token_time(
            [
                {
                    "type": "text-chunks",
                    "time0": 1000,
                    "dt": [50],
                    "texts": ["", ""],
                    "index": 0,
                }
            ]
        )
        is None
    )
    # A run without dt at all: members share time0.
    assert (
        first_token_time(
            [{"type": "text-chunks", "time0": 1000, "texts": ["", "a"], "index": 0}]
        )
        == 1000.0
    )
    # Short dt: a missing gap counts as zero.
    assert (
        first_token_time(
            [
                {
                    "type": "text-chunks",
                    "time0": 1000,
                    "dt": [],
                    "texts": ["", "a"],
                    "index": 0,
                }
            ]
        )
        == 1000.0
    )
    # Non-numeric gap counts as zero.
    assert (
        first_token_time(
            [
                {
                    "type": "text-chunks",
                    "time0": 1000,
                    "dt": ["x"],
                    "texts": ["", "a"],
                    "index": 0,
                }
            ]
        )
        == 1000.0
    )
    # Tool-call run with a name starts at time0 immediately.
    assert (
        first_token_time(
            [
                {
                    "type": "tool-call-chunks",
                    "time0": 1000,
                    "dt": [],
                    "args": [""],
                    "index": 0,
                    "name": "add",
                }
            ]
        )
        == 1000.0
    )
    # Tool-call run without a name takes the first non-empty argument.
    assert (
        first_token_time(
            [
                {
                    "type": "tool-call-chunks",
                    "time0": 1000,
                    "dt": [40],
                    "args": ["", '{"a":1}'],
                    "index": 0,
                }
            ]
        )
        == 1040.0
    )
    # Missing fragments / missing time0 qualify nothing.
    assert (
        first_token_time([{"type": "text-chunks", "time0": 1000, "dt": [], "index": 0}])
        is None
    )
    assert (
        first_token_time(
            [{"type": "text-chunks", "dt": [], "texts": ["a"], "index": 0}]
        )
        is None
    )
    assert (
        first_token_time(
            [{"type": "tool-call-chunks", "dt": [], "args": [""], "index": 0}]
        )
        is None
    )
    # Scanning stops at the first qualifying record.
    assert (
        first_token_time(
            [
                {
                    "type": "chunk",
                    "time": 900,
                    "chunk": {"type": "usage", "outputTokens": 1},
                },
                {
                    "type": "chunk",
                    "time": 950,
                    "chunk": {"type": "text-delta", "index": 0, "text": "t"},
                },
            ]
        )
        == 950.0
    )


def test_output_tokens_guards() -> None:
    """Usage output tokens are accepted only as a finite number >= 0."""
    assert _output_tokens({"outputTokens": 5}) == 5
    assert _output_tokens({"outputTokens": 0}) == 0
    assert _output_tokens(None) is None
    assert _output_tokens("usage") is None
    assert _output_tokens({}) is None
    assert _output_tokens({"outputTokens": True}) is None
    assert _output_tokens({"outputTokens": -1}) is None
    assert _output_tokens({"outputTokens": float("nan")}) is None
    assert _output_tokens({"outputTokens": "5"}) is None


def test_session_stats_state_schema() -> None:
    """The stats state schema enforces totals plus in-flight boundaries."""
    valid = SESSION_STATS_PROJECTION.init(HEADER, 0)
    assert validate_session_stats_state(valid) == {
        **valid,
        "llm_ms": 0.0,
        "tool_ms": 0.0,
        "ttft_ms": 0.0,
        "decode_ms": 0.0,
    }
    for bad in (
        {**valid, "turns": -1},
        {**valid, "turns": True},
        {**valid, "llm_ms": -1.0},
        {**valid, "llm_ms": float("nan")},
        {**valid, "last_turn": -1},
        {**valid, "open_step": {"turn": 1, "step": 1, "start_time": 5.0}},
        {**valid, "pending_calls": {"c1": -2.0}},
        {**valid, "extra": 0},
        {k: v for k, v in valid.items() if k != "steps"},
    ):
        with pytest.raises(ValueError):
            validate_session_stats_state(bad)


# ---------------------------------------------------------------------------
# preview
# ---------------------------------------------------------------------------


def test_preview_budgets_and_ellipsis() -> None:
    """Previews join text blocks, collapse whitespace, and cap with ellipsis."""
    assert preview([{"type": "text", "text": "hello"}], 50) == "hello"
    assert preview([], 50) == ""
    assert preview([{"type": "image", "text": "nope"}], 50) == ""
    assert preview([None], 50) == ""  # type: ignore[list-item]
    assert preview([{"type": "text", "text": 5}], 50) == ""  # type: ignore[dict-item]
    assert preview([{"type": "text"}], 50) == ""
    # Unread remainder with a short normalized text still earns an ellipsis.
    assert (
        preview(
            [{"type": "text", "text": " " * 100}, {"type": "text", "text": "more"}],
            50,
        )
        == "…"
    )
    assert (
        preview(
            [
                {"type": "text", "text": "  hello   world  "},
                {"type": "text", "text": "again"},
            ],
            50,
        )
        == "hello world again"
    )
    long_prompt = "word " * 20
    clipped = preview([{"type": "text", "text": long_prompt}], 50)
    assert len(clipped) <= 50
    assert clipped.endswith("…")
    assert clipped == long_prompt.strip()[:49].rstrip() + "…"
    # A single multi-megabyte block is clipped at twice the limit first.
    huge = "x" * 500
    assert preview([{"type": "text", "text": huge}], 50) == "x" * 49 + "…"
    # Unread remainder without a length clip still earns an ellipsis.
    blocks = [
        {"type": "text", "text": "a" * 100},
        {"type": "text", "text": "b" * 100},
    ]
    assert preview(blocks, 50) == "a" * 49 + "…"
    # A cut mid-word leaves no trailing space before the ellipsis.
    spaced = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj kkkk"
    assert preview([{"type": "text", "text": spaced}], 50).endswith("…")
    # Exactly at the budget: no ellipsis.
    exact = "x" * 49
    assert preview([{"type": "text", "text": exact}], 50) == exact


def test_preview_counts_code_points() -> None:
    """Astral-plane characters count one each (accepted divergence)."""
    assert preview([{"type": "text", "text": "😀" * 60}], 50) == "😀" * 49 + "…"


# ---------------------------------------------------------------------------
# turnOutline
# ---------------------------------------------------------------------------


def _outline_view(events: list[SessionEvent]) -> list[dict[str, Any]]:
    state = _fold(TURN_OUTLINE_PROJECTION, events)
    assert TURN_OUTLINE_PROJECTION.wire is not None
    return TURN_OUTLINE_PROJECTION.wire.view(state)


def test_turn_outline_scenario_c_retried_turn_and_steering() -> None:
    """One entry per turn: retried boundaries and steering keep the first."""
    events = [
        _event("turn/start", {"turn": 1}, 0, 1000),
        _user_message("first prompt", 1, 1100),
        _event("turn/start", {"turn": 1}, 2, 1200),
        _user_message("steering prompt", 3, 1300),
        _assistant_message("draft one", 4, 1400),
        _event("turn/end", {"turn": 1}, 5, 1500),
    ]
    assert _outline_view(events) == [
        {"turn": 1, "seq": 0, "prompt": "first prompt", "response": "draft one"}
    ]


def test_turn_outline_response_commits_only_at_turn_end() -> None:
    """A turn with no assistant text keeps an empty response; drafts wait."""
    events = [
        _event("turn/start", {"turn": 1}, 0, 1000),
        _user_message("ask", 1, 1100),
        _event("turn/end", {"turn": 1}, 2, 1200),
        _assistant_message("late draft", 3, 1300),
        _event("turn/start", {"turn": 2}, 4, 1400),
        _event("turn/end", {"turn": 2}, 5, 1500),
    ]
    # The draft that arrives after turn 1 ended is discarded by turn 2's
    # boundary: a turn with no assistant text keeps an empty response.
    assert _outline_view(events) == [
        {"turn": 1, "seq": 0, "prompt": "ask", "response": ""},
        {"turn": 2, "seq": 4, "prompt": "", "response": ""},
    ]
    # A draft that arrives inside the open turn commits at its turn/end.
    events2 = [
        _event("turn/start", {"turn": 1}, 0, 1000),
        _assistant_message("inside", 1, 1100),
        _event("turn/end", {"turn": 1}, 2, 1200),
    ]
    assert _outline_view(events2) == [
        {"turn": 1, "seq": 0, "prompt": "", "response": "inside"}
    ]


def test_turn_outline_draft_only_applies_keep_turns_identity() -> None:
    """Draft-only applies reuse the turns list; turn/end commits from it."""
    state = TURN_OUTLINE_PROJECTION.init(HEADER, 0)
    state = TURN_OUTLINE_PROJECTION.apply(
        state, _event("turn/start", {"turn": 1}, 0, 1000)
    )
    turns = state["turns"]
    drafted = TURN_OUTLINE_PROJECTION.apply(state, _assistant_message("hello", 1, 1100))
    assert drafted is not state
    assert drafted["turns"] is turns
    assert drafted["draft"] == "hello"
    # Same draft again is a no-op; an empty draft is a no-op.
    assert (
        TURN_OUTLINE_PROJECTION.apply(drafted, _assistant_message("hello", 2, 1200))
        is drafted
    )
    assert (
        TURN_OUTLINE_PROJECTION.apply(drafted, _assistant_message("", 2, 1200))
        is drafted
    )
    # Committing a new draft replaces the standing entry (new list).
    committed = TURN_OUTLINE_PROJECTION.apply(
        drafted, _event("turn/end", {"turn": 1}, 3, 1300)
    )
    assert committed["turns"] is not turns
    assert committed["turns"][0]["response"] == "hello"
    assert committed["draft"] == ""
    again = TURN_OUTLINE_PROJECTION.apply(
        committed, _assistant_message("hello", 4, 1400)
    )
    cleared = TURN_OUTLINE_PROJECTION.apply(
        again, _event("turn/end", {"turn": 1}, 5, 1500)
    )
    # A draft equal to the standing response clears the draft and keeps
    # the entry list identical.
    assert cleared["turns"] is committed["turns"]
    assert cleared["draft"] == ""


def test_turn_outline_ignores_events_without_a_turn() -> None:
    """Prompts need a turn entry; assistant drafts still buffer without one."""
    state = TURN_OUTLINE_PROJECTION.init(HEADER, 0)
    assert TURN_OUTLINE_PROJECTION.apply(state, _user_message("hi", 0, 1000)) is state
    drafted = TURN_OUTLINE_PROJECTION.apply(
        state, _assistant_message("orphan", 0, 1000)
    )
    assert drafted["draft"] == "orphan"
    ended = TURN_OUTLINE_PROJECTION.apply(
        drafted, _event("turn/end", {"turn": 1}, 1, 1100)
    )
    assert ended == {"turns": [], "draft": ""}


def test_turn_outline_ignores_non_user_prompts_and_empty_prompts() -> None:
    """Synthetic messages and empty prompts never fill the prompt slot."""
    state = TURN_OUTLINE_PROJECTION.init(HEADER, 0)
    state = TURN_OUTLINE_PROJECTION.apply(
        state, _event("turn/start", {"turn": 1}, 0, 1000)
    )
    synthetic = _event(
        "user/message",
        {
            "content": [{"type": "text", "text": "injected"}],
            "source": {"kind": "plugin:compact"},
        },
        1,
        1100,
        surface_op="append",
    )
    assert TURN_OUTLINE_PROJECTION.apply(state, synthetic) is state
    assert TURN_OUTLINE_PROJECTION.apply(state, _user_message("   ", 1, 1100)) is state
    assert (
        TURN_OUTLINE_PROJECTION.apply(
            state,
            _event("user/message", {"source": {"kind": "user"}}, 1, 1100),
        )
        is state
    )
    assert (
        TURN_OUTLINE_PROJECTION.apply(
            state, _event("turn/start", {"turn": "one"}, 1, 1100)
        )
        is state
    )
    assert (
        TURN_OUTLINE_PROJECTION.apply(
            state,
            _event("assistant/message", {"message": {}}, 1, 1100),
        )
        is state
    )


def test_turn_outline_state_schema() -> None:
    """Outline state enforces entry budgets and strictly increasing turns."""
    valid = {
        "turns": [{"turn": 1, "seq": 0, "prompt": "p", "response": "r"}],
        "draft": "",
    }
    assert validate_turn_outline_state(valid) == valid
    assert validate_turn_outline_view(valid["turns"]) == valid["turns"]
    for bad in (
        {"turns": [], "draft": "x" * 121},
        {
            "turns": [
                {"turn": 2, "seq": 0, "prompt": "", "response": ""},
                {"turn": 1, "seq": 5, "prompt": "", "response": ""},
            ],
            "draft": "",
        },
        {
            "turns": [
                {"turn": 1, "seq": 0, "prompt": "", "response": ""},
                {"turn": 1, "seq": 5, "prompt": "", "response": ""},
            ],
            "draft": "",
        },
        {
            "turns": [{"turn": 1, "seq": 0, "prompt": "x" * 51, "response": ""}],
            "draft": "",
        },
        {
            "turns": [{"turn": 1, "seq": 0, "prompt": "", "response": "x" * 121}],
            "draft": "",
        },
        {"turns": [{"turn": 1, "seq": 0, "prompt": ""}], "draft": ""},
        {"turns": [], "draft": "", "extra": 1},
    ):
        with pytest.raises(ValueError):
            validate_turn_outline_state(bad)
    with pytest.raises(ValueError):
        validate_turn_outline_view(
            [
                {"turn": 2, "seq": 0, "prompt": "", "response": ""},
                {"turn": 2, "seq": 5, "prompt": "", "response": ""},
            ]
        )


# ---------------------------------------------------------------------------
# workingDirectory
# ---------------------------------------------------------------------------


def test_working_directory_init_and_fold() -> None:
    """Init starts at the header cwd; only change events move it."""
    assert WORKING_DIRECTORY_PROJECTION.init(HEADER, 0) == "/home/malcom/work"
    assert WORKING_DIRECTORY_PROJECTION.init({"version": 4, "id": "s"}, 0) is None
    assert (
        WORKING_DIRECTORY_PROJECTION.init({"version": 4, "id": "s", "cwd": ""}, 0)
        is None
    )
    assert (
        WORKING_DIRECTORY_PROJECTION.init({"version": 4, "id": "s", "cwd": 7}, 0)
        is None
    )
    state = WORKING_DIRECTORY_PROJECTION.init(HEADER, 0)
    assert (
        WORKING_DIRECTORY_PROJECTION.apply(
            state, _event("turn/start", {"turn": 1}, 0, 1000)
        )
        is state
    )
    assert (
        WORKING_DIRECTORY_PROJECTION.apply(
            state, _event("working-directory/change", {"cwd": "/tmp/proj"}, 0, 1000)
        )
        == "/tmp/proj"
    )
    assert (
        WORKING_DIRECTORY_PROJECTION.apply(
            state, _event("working-directory/change", {}, 0, 1000)
        )
        is state
    )


def test_working_directory_value_schema() -> None:
    """The directory schema takes an absolute path or None."""
    assert validate_working_directory_value("/tmp/x") == "/tmp/x"
    assert validate_working_directory_value(None) is None
    for bad in ("", "relative/path", 5, True):
        with pytest.raises(ValueError):
            validate_working_directory_value(bad)


# ---------------------------------------------------------------------------
# Same-reference discipline and init freshness.
# ---------------------------------------------------------------------------


def test_uninterested_events_return_the_same_state_reference() -> None:
    """Every unit returns the identical object for an event it ignores."""
    uninterested = _event("approval/policy", {"policy": "x"}, 0, 1000)
    for definition in ALL_DEFINITIONS:
        state = definition.init(HEADER, 0)
        assert definition.apply(state, uninterested) is state


def test_init_returns_fresh_objects() -> None:
    """Init never aliases a shared constant: each call is a fresh object."""
    first = TITLE_INPUT_PROJECTION.init(HEADER, 0)
    second = TITLE_INPUT_PROJECTION.init(HEADER, 0)
    assert first == second and first is not second
    first_outline = TURN_OUTLINE_PROJECTION.init(HEADER, 0)
    second_outline = TURN_OUTLINE_PROJECTION.init(HEADER, 0)
    assert first_outline == second_outline
    assert first_outline is not second_outline
    assert first_outline["turns"] is not second_outline["turns"]
    first_stats = SESSION_STATS_PROJECTION.init(HEADER, 0)
    second_stats = SESSION_STATS_PROJECTION.init(HEADER, 0)
    assert first_stats["pending_calls"] is not second_stats["pending_calls"]
    assert TITLE_PROJECTION.state_version == 1
    assert SESSION_STATS_PROJECTION.state_version == 1
    assert TURN_OUTLINE_PROJECTION.state_version == 2
    assert WORKING_DIRECTORY_PROJECTION.state_version == 1


# ---------------------------------------------------------------------------
# Service wiring.
# ---------------------------------------------------------------------------


def test_rows_wire_the_five_units() -> None:
    """The bundle rows expose the registry with all five units driving."""
    from magitech_harness import create_harness

    ctx = create_harness()
    service = ctx.sessions
    registry = ctx.sessionProjections
    assert registry.summary_keys() == frozenset({"title", "workingDirectory"})
    session = service.create("svc-1")
    service.log_event(session, "turn/start", {"turn": 1})
    service.log_event(
        session,
        "user/message",
        {
            "content": [{"type": "text", "text": "Hello"}],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )
    snapshot = registry.snapshot(session)
    assert snapshot["values"]["turnOutline"] == [
        {"turn": 1, "seq": 0, "prompt": "Hello", "response": ""}
    ]
    assert registry.checkpoint(session)["titleInput"]["val"]["count"] == 1
