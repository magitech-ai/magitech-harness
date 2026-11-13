"""Unit tests for the surface fold (fold_surface / SurfaceManager)."""

import pytest

from magitech_harness.session.events import SessionEvent
from magitech_harness.session.surface import (
    MessageProjection,
    SurfaceError,
    SurfaceManager,
    derive_event_message,
    fold_surface,
)


def _event(
    seq: int,
    type: str,
    payload: dict | None = None,
    **kwargs: object,
) -> SessionEvent:
    return SessionEvent(
        id=f"s:{seq}",
        session_id="s",
        type=type,
        timestamp=1000.0 + seq,
        payload=payload or {},
        seq=seq,
        **kwargs,  # type: ignore[arg-type]
    )


def _user(seq: int, text: str = "hi", **kwargs: object) -> SessionEvent:
    return _event(
        seq,
        "user/message",
        {"role": "user", "id": f"m{seq}", "content": [{"type": "text", "text": text}]},
        surface_op=kwargs.pop("surface_op", "append"),
        **kwargs,
    )


def _system(seq: int, text: str = "Be helpful.", **kwargs: object) -> SessionEvent:
    return _event(
        seq,
        "system/message",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": f"sys{seq}",
                "role": "system",
                "content": [{"type": "text", "text": text}],
            },
        },
        surface_op=kwargs.pop("surface_op", "append"),
        **kwargs,
    )


def _assistant(seq: int, text: str = "done", **kwargs: object) -> SessionEvent:
    return _event(
        seq,
        "assistant/message",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": f"a{seq}",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
            "stream": [],
        },
        surface_op=kwargs.pop("surface_op", "append"),
        **kwargs,
    )


def _tool_result(
    seq: int, call_id: str = "c1", text: str = "3", **kwargs: object
) -> SessionEvent:
    return _event(
        seq,
        "tool/result",
        {
            "turn": 1,
            "step": 1,
            "message": {
                "id": f"t{seq}",
                "role": "tool",
                "toolCallId": call_id,
                "isError": False,
                "source": {"kind": "tool", "callId": call_id},
                "content": [{"type": "text", "text": text}],
            },
        },
        surface_op=kwargs.pop("surface_op", "append"),
        **kwargs,
    )


def test_fold_empty_log() -> None:
    """Verifies an empty log folds to an empty surface."""
    result = fold_surface([])
    assert result.nodes == []
    assert result.replacements == []
    assert result.projected_messages == {}


def test_fold_appends_in_order_and_ignores_log_only() -> None:
    """Verifies surface nodes follow log order and log-only events stay out."""
    events = [
        _event(0, "turn/start", {"turn": 1}),
        _system(1),
        _user(2),
        _assistant(3),
        _event(4, "step/end", {"turn": 1, "step": 1}),
        _tool_result(5),
        _event(6, "turn/end", {"turn": 1, "reason": {"kind": "completed"}}),
    ]
    result = fold_surface(events)
    assert result.nodes == [1, 2, 3, 5]
    assert result.replacements == []


def test_fold_surface_type_without_marker_raises() -> None:
    """Verifies a surface-eligible event without surfaceOp is refused."""
    with pytest.raises(SurfaceError, match="requires a surfaceOp marker"):
        fold_surface([_event(0, "user/message", {"text": "hi"})])


def test_fold_non_surface_type_with_marker_raises() -> None:
    """Verifies a log-only event carrying surfaceOp is refused."""
    with pytest.raises(SurfaceError, match="not surface-eligible"):
        fold_surface([_event(0, "turn/start", {"turn": 1}, surface_op="append")])
    with pytest.raises(SurfaceError, match="not surface-eligible"):
        fold_surface([_event(0, "turn/start", {"turn": 1}, source_event_seqs=[0])])


def test_fold_assistant_message_cannot_carry_source_seqs() -> None:
    """Verifies assistant/message embeds its stream and cites no sources."""
    with pytest.raises(SurfaceError, match="cannot carry sourceEventSeqs"):
        fold_surface([_assistant(0, source_event_seqs=[0])])


def test_fold_source_seqs_must_be_earlier_unique_nonempty() -> None:
    """Verifies sourceEventSeqs attribution rules on a replacement."""
    base = [_system(0), _user(1)]
    replace = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 1},
        source_event_seqs=[],
    )
    with pytest.raises(SurfaceError, match="must not be empty"):
        fold_surface([*base, replace])
    replace = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 1},
        source_event_seqs=[1, 1],
    )
    with pytest.raises(SurfaceError, match="must not contain duplicates"):
        fold_surface([*base, replace])
    replace = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 1},
        source_event_seqs=[2],
    )
    with pytest.raises(SurfaceError, match="must reference earlier events"):
        fold_surface([*base, replace])


def test_fold_replace_shadows_range_and_records_replacement() -> None:
    """Verifies a replace shadows its range and records the operation."""
    events = [_system(0), _user(1), _user(2)]
    replacement = _user(
        3,
        "replacement",
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 2},
        source_event_seqs=[1, 2],
    )
    result = fold_surface([*events, replacement])
    assert result.nodes == [0, 3]
    assert len(result.replacements) == 1
    assert result.replacements[0].seq == 3
    assert result.replacements[0].start == 1
    assert result.replacements[0].end == 2
    assert result.replacements[0].shadowed_seqs == [1, 2]


def test_fold_replace_must_cite_every_shadowed_node() -> None:
    """Verifies a replace citing only part of its range is refused."""
    events = [_system(0), _user(1), _user(2)]
    replacement = _user(
        3,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 2},
        source_event_seqs=[1],
    )
    with pytest.raises(SurfaceError, match="must include every shadowed"):
        fold_surface([*events, replacement])


def test_fold_replace_range_must_exist_in_surface() -> None:
    """Verifies a replace over a missing or inverted range is refused."""
    events = [_system(0), _user(1)]
    missing = _user(
        2,
        surface_op={"op": "replace", "startSeq": 5, "endSeq": 5},
        source_event_seqs=[5],
    )
    with pytest.raises(SurfaceError, match="must reference earlier events"):
        fold_surface([*events, missing])
    inverted = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 0},
        source_event_seqs=[0, 1],
    )
    with pytest.raises(SurfaceError, match="is after end seq"):
        fold_surface([*events, inverted])


def test_fold_replace_with_future_endpoint_raises() -> None:
    """Verifies a replace endpoint at or after the replacing event is refused."""
    events = [_system(0), _user(1)]
    future = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 2},
        source_event_seqs=[1, 2],
    )
    with pytest.raises(SurfaceError, match="must reference earlier events"):
        fold_surface([*events, future])


def test_fold_tool_result_replace_may_change_only_content() -> None:
    """Verifies a tool/result replace rewrites exactly one node, content only."""
    events = [_system(0), _user(1), _tool_result(2, text="3")]
    replacement = _tool_result(3, text="4")
    replacement.payload["message"]["id"] = "t2"
    replacement.surface_op = {"op": "replace", "startSeq": 2, "endSeq": 2}
    replacement.source_event_seqs = [2]
    result = fold_surface([*events, replacement])
    assert result.nodes == [0, 1, 3]

    wrong_turn = _tool_result(3, text="4")
    wrong_turn.payload["message"]["id"] = "t2"
    wrong_turn.payload = {**wrong_turn.payload, "turn": 2}
    wrong_turn.surface_op = {"op": "replace", "startSeq": 2, "endSeq": 2}
    wrong_turn.source_event_seqs = [2]
    with pytest.raises(SurfaceError, match="may change only content"):
        fold_surface([*events, wrong_turn])

    two_nodes = _tool_result(3, text="4")
    two_nodes.surface_op = {"op": "replace", "startSeq": 1, "endSeq": 2}
    two_nodes.source_event_seqs = [1, 2]
    with pytest.raises(SurfaceError, match="exactly one current node"):
        fold_surface([*events, two_nodes])

    wrong_target = _tool_result(3, text="4")
    wrong_target.surface_op = {"op": "replace", "startSeq": 1, "endSeq": 1}
    wrong_target.source_event_seqs = [1]
    with pytest.raises(SurfaceError, match="must target a current tool/result"):
        fold_surface([*events, wrong_target])


def test_fold_system_head_protection() -> None:
    """Verifies node 0's system prompt is rewritten only by a system node."""
    events = [_system(0), _user(1)]
    by_user = _user(
        2,
        surface_op={"op": "replace", "startSeq": 0, "endSeq": 0},
        source_event_seqs=[0],
    )
    with pytest.raises(SurfaceError, match="node 0 holds the system prompt"):
        fold_surface([*events, by_user])
    by_system_two = _system(2)
    by_system_two.surface_op = {"op": "replace", "startSeq": 0, "endSeq": 1}
    by_system_two.source_event_seqs = [0, 1]
    with pytest.raises(SurfaceError, match="node 0 holds the system prompt"):
        fold_surface([*events, by_system_two])
    by_system = _system(2, "Updated.")
    by_system.surface_op = {"op": "replace", "startSeq": 0, "endSeq": 0}
    by_system.source_event_seqs = [0]
    result = fold_surface([*events, by_system])
    assert result.nodes == [2, 1]


def test_fold_seq_gap_raises() -> None:
    """Verifies a non-contiguous log is refused by the fold."""
    with pytest.raises(SurfaceError, match="is not contiguous"):
        fold_surface([_user(0), _user(2)])


def test_fold_unknown_ignorable_keeps_surface() -> None:
    """Verifies an unknown ignorable event never changes the surface."""
    events = [
        _user(0),
        _event(1, "vendor/widget", {"x": 1}, ignorable=True),
        _user(2),
    ]
    result = fold_surface(events)
    assert result.nodes == [0, 2]


def test_fold_unknown_non_ignorable_with_surface_op_raises() -> None:
    """Verifies an unknown event carrying surface metadata is refused."""
    events = [_event(0, "vendor/widget", {"x": 1}, surface_op="append")]
    with pytest.raises(SurfaceError, match="not surface-eligible"):
        fold_surface(events)


class _ImageOffloadProjection(MessageProjection):
    """Test projection interpreting image/offload decisions."""

    type = "image/offload"

    def project(self, event: SessionEvent, context: object) -> dict[int, dict]:
        return {0: {"role": "user", "content": "image offloaded"}}


def test_fold_image_offload_requires_projection() -> None:
    """Verifies image/offload throws without a registered message projection."""
    event = _event(0, "image/offload", {"seq": 0})
    with pytest.raises(SurfaceError, match="requires a message projection"):
        fold_surface([event])
    result = fold_surface([event], projections=[_ImageOffloadProjection()])
    assert result.nodes == []
    assert result.projected_messages[0]["content"] == "image offloaded"


def _developer(
    seq: int, blocks: list[dict], header_seq: int | None = None
) -> SessionEvent:
    payload: dict = {
        "turn": 1,
        "step": 1,
        "message": {
            "id": f"d{seq}",
            "role": "developer",
            "content": blocks,
        },
    }
    if header_seq is not None:
        payload["headerSeq"] = header_seq
    return _event(seq, "developer/message", payload, surface_op="append")


def _header(seq: int, tools: list[dict]) -> SessionEvent:
    return _event(
        seq,
        "request/header",
        {"header": {"config": {}, "tools": tools}, "reason": {"kind": "initial"}},
    )


def test_fold_developer_header_seq_rules() -> None:
    """Verifies tool additions resolve against their named header."""
    tools = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    addition = {"type": "tool-addition", "toolName": "search"}
    events = [_header(0, tools), _developer(1, [addition], header_seq=0)]
    assert fold_surface(events).nodes == [1]

    # Addition without headerSeq.
    with pytest.raises(SurfaceError, match="requires headerSeq exactly"):
        fold_surface([_developer(0, [addition])])
    # headerSeq without additions.
    with pytest.raises(SurfaceError, match="requires headerSeq exactly"):
        fold_surface([_header(0, tools), _developer(1, [], header_seq=0)])
    # headerSeq naming a non-header event.
    with pytest.raises(SurfaceError, match="must reference an earlier request/header"):
        fold_surface([_user(0), _developer(1, [addition], header_seq=0)])
    # Addition naming an unknown tool.
    with pytest.raises(SurfaceError, match="must name exactly one tool"):
        fold_surface(
            [
                _header(0, tools),
                _developer(
                    1, [{"type": "tool-addition", "toolName": "nope"}], header_seq=0
                ),
            ]
        )
    # Addition naming a tool with an incomplete definition.
    with pytest.raises(SurfaceError, match="requires a complete tool definition"):
        fold_surface(
            [
                _header(0, [{"name": "search"}]),
                _developer(1, [addition], header_seq=0),
            ]
        )
    # Referenced tool with deferLoading False.
    bad_tools = [dict(tools[0], deferLoading=False)]
    with pytest.raises(SurfaceError, match="deferLoading must be true"):
        fold_surface([_header(0, bad_tools), _developer(1, [addition], header_seq=0)])
    # Inline tool definition on an addition.
    with pytest.raises(SurfaceError, match="must omit inline tool definitions"):
        fold_surface(
            [
                _header(0, tools),
                _developer(1, [dict(addition, tool={"name": "search"})], header_seq=0),
            ]
        )


def test_validate_session_event_data_rules() -> None:
    """Verifies payload rules: developer role, request header, tool error."""
    from magitech_harness.session.surface import validate_session_event_data

    # Developer message without developer role.
    bad_dev = _event(
        0,
        "developer/message",
        {
            "message": {
                "id": "d0",
                "role": "user",
                "content": [{"type": "text", "text": "x"}],
            }
        },
        surface_op="append",
    )
    with pytest.raises(SurfaceError, match="must occur together"):
        validate_session_event_data(bad_dev, "test")
    # Tool-change blocks require developer role.
    bad_role = _event(
        0,
        "user/message",
        {
            "role": "user",
            "id": "m0",
            "content": [{"type": "tool-addition", "toolName": "search"}],
        },
        surface_op="append",
    )
    with pytest.raises(SurfaceError, match="tool-change blocks require"):
        validate_session_event_data(bad_role, "test")
    # Developer message with a plugin-owned source.
    plugin_dev = _developer(
        0,
        [{"type": "text", "text": "x"}],
    )
    plugin_dev.payload["message"]["source"] = {"kind": "plugin"}
    with pytest.raises(SurfaceError, match="must not be plugin-owned"):
        validate_session_event_data(plugin_dev, "test")
    # request/header with a system field.
    bad_header = _event(
        0,
        "request/header",
        {"header": {"system": "prompt", "config": {}}, "reason": {"kind": "initial"}},
    )
    with pytest.raises(SurfaceError, match="must omit header.system"):
        validate_session_event_data(bad_header, "test")
    # request/header with empty tools.
    empty_tools = _event(
        0, "request/header", {"header": {"config": {}, "tools": []}, "reason": {}}
    )
    with pytest.raises(SurfaceError, match="must omit empty tools"):
        validate_session_event_data(empty_tools, "test")
    # request/header with empty adapterDefaults.
    empty_defaults = _event(
        0,
        "request/header",
        {"header": {"config": {}, "adapterDefaults": {}}, "reason": {}},
    )
    with pytest.raises(SurfaceError, match="must omit empty adapterDefaults"):
        validate_session_event_data(empty_defaults, "test")
    # tool/result error without message.isError.
    bad_result = _event(
        0,
        "tool/result",
        {"message": {"role": "tool", "isError": False}, "error": {"code": "X"}},
        surface_op="append",
    )
    with pytest.raises(SurfaceError, match="error requires message.isError"):
        validate_session_event_data(bad_result, "test")
    # System message carrying a tool-result block.
    bad_system = _system(0)
    bad_system.payload["message"]["content"] = [
        {"type": "tool-result", "toolCallId": "c1"}
    ]
    with pytest.raises(SurfaceError, match="must not carry tool-result blocks"):
        validate_session_event_data(bad_system, "test")


def test_derive_event_message_rules() -> None:
    """Verifies single-event projection: verbatim user, empty content null."""
    assert derive_event_message(_user(0, "hello")) == {
        "role": "user",
        "id": "m0",
        "content": [{"type": "text", "text": "hello"}],
    }
    assert derive_event_message(_event(0, "turn/start", {"turn": 1})) is None
    empty_assistant = _assistant(0)
    empty_assistant.payload["message"]["content"] = []
    assert derive_event_message(empty_assistant) is None
    assert derive_event_message(_assistant(0)) == {
        "id": "a0",
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
    }
    assert derive_event_message(_system(0)) == {
        "id": "sys0",
        "role": "system",
        "content": [{"type": "text", "text": "Be helpful."}],
    }
    assert derive_event_message(_tool_result(0))["role"] == "tool"
    projected = derive_event_message(_user(0), {0: {"role": "user", "content": "p"}})
    assert projected == {"role": "user", "content": "p"}


def test_surface_manager_validate_next_is_atomic() -> None:
    """Verifies a failed candidate cannot partially mutate the surface."""
    log = [_system(0), _user(1)]
    manager = SurfaceManager(log)
    assert manager.nodes == [0, 1]
    assert manager.replace_generation == 0
    bad = _user(
        2,
        surface_op={"op": "replace", "startSeq": 9, "endSeq": 9},
        source_event_seqs=[9],
    )
    with pytest.raises(SurfaceError, match="must reference earlier events"):
        manager.validate_next(bad)
    assert manager.nodes == [0, 1]
    assert manager.replace_generation == 0
    good = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 1},
        source_event_seqs=[1],
    )
    manager.validate_next(good)
    # Still uncommitted until the event enters the log.
    assert manager.nodes == [0, 1]
    log.append(good)
    assert manager.nodes == [0, 2]
    assert manager.replace_generation == 1
    assert manager.content_generation == 1


def test_surface_manager_projection_removal_invalidates() -> None:
    """Verifies removing a used projection refuses further derivation."""
    projections: list = [_ImageOffloadProjection()]
    log = [_event(0, "image/offload", {"seq": 0})]
    manager = SurfaceManager(log, projections)
    manager.validate_next(_event(1, "turn/start", {"turn": 1}))
    # The projection was used by the fold of seq 0; removing it invalidates.
    projections.clear()
    with pytest.raises(SurfaceError, match="was removed or replaced"):
        _ = manager.nodes
    with pytest.raises(SurfaceError, match="was removed or replaced"):
        manager.validate_next(_event(1, "turn/start", {"turn": 1}))


def test_surface_manager_derives_with_projections() -> None:
    """Verifies derive_event_message applies committed projections."""
    projections: list = [_ImageOffloadProjection()]
    log = [_event(0, "image/offload", {"seq": 0}), _user(1)]
    manager = SurfaceManager(log, projections)
    assert manager.derive_event_message(log[0]) == {
        "role": "user",
        "content": "image offloaded",
    }
    assert manager.derive_event_message(log[1])["role"] == "user"


def test_validate_tool_addition_requires_tool_name() -> None:
    """Verifies a tool-addition without a nonempty toolName is refused."""
    from magitech_harness.session.surface import validate_session_event_data

    event = _developer(0, [{"type": "tool-addition"}])
    with pytest.raises(SurfaceError, match="requires a nonempty toolName"):
        validate_session_event_data(event, "test")


def test_validate_system_role_must_be_system() -> None:
    """Verifies a system/message whose role is not system is refused."""
    from magitech_harness.session.surface import validate_session_event_data

    event = _system(0)
    event.payload["message"]["role"] = "user"
    with pytest.raises(SurfaceError, match="system/message role must be system"):
        validate_session_event_data(event, "test")


def test_validate_request_header_header_must_be_object() -> None:
    """Verifies a request/header whose header is not an object is refused."""
    from magitech_harness.session.surface import validate_session_event_data

    event = _event(0, "request/header", {"header": "not-an-object", "reason": {}})
    with pytest.raises(SurfaceError, match="header must be an object"):
        validate_session_event_data(event, "test")


def test_validate_tool_result_error_with_is_error_is_valid() -> None:
    """Verifies a tool/result error with message.isError true is valid."""
    from magitech_harness.session.surface import validate_session_event_data

    event = _tool_result(0)
    event.payload["error"] = {"code": "X"}
    event.payload["message"]["isError"] = True
    validate_session_event_data(event, "test")


def test_fold_invalid_surface_op() -> None:
    """Verifies a surfaceOp that is neither append nor replace is refused."""
    with pytest.raises(SurfaceError, match="carries an invalid surfaceOp"):
        fold_surface([_user(0, surface_op="bogus")])


def test_fold_source_seq_must_be_event_seq() -> None:
    """Verifies sourceEventSeqs must densely contain event seqs."""
    replacement = _user(
        2,
        surface_op={"op": "replace", "startSeq": 0, "endSeq": 0},
        source_event_seqs=[-1],
    )
    with pytest.raises(SurfaceError, match="must densely contain"):
        fold_surface([_user(0), _user(1), replacement])


def test_fold_developer_header_seq_must_be_earlier() -> None:
    """Verifies headerSeq must reference an earlier request/header."""
    tools = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    addition = {"type": "tool-addition", "toolName": "search"}
    # headerSeq equal to the developer event's own seq is not earlier.
    with pytest.raises(SurfaceError, match="must reference an earlier"):
        fold_surface([_header(0, tools), _developer(1, [addition], header_seq=1)])


def test_fold_developer_with_text_block_and_header_seq() -> None:
    """Verifies non-addition content blocks are skipped in header checks."""
    tools = [
        {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    blocks = [
        {"type": "text", "text": "note"},
        {"type": "tool-addition", "toolName": "search"},
    ]
    result = fold_surface([_header(0, tools), _developer(1, blocks, header_seq=0)])
    assert result.nodes == [1]


def test_fold_tool_removal_block_needs_no_header_seq() -> None:
    """Verifies a tool-removal block does not count as a tool addition."""
    blocks = [{"type": "tool-removal", "toolName": "search"}]
    result = fold_surface([_developer(0, blocks)])
    assert result.nodes == [0]


def test_fold_replace_start_not_in_surface() -> None:
    """Verifies a replace whose start seq is not a surface node is refused."""
    events = [_user(0), _event(1, "turn/start", {"turn": 1})]
    replacement = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 1},
        source_event_seqs=[1],
    )
    with pytest.raises(SurfaceError, match="start seq 1 not found in surface"):
        fold_surface([*events, replacement])


def test_fold_replace_end_not_in_surface() -> None:
    """Verifies a replace whose end seq is not a surface node is refused."""
    events = [_user(0), _event(1, "turn/start", {"turn": 1})]
    replacement = _user(
        2,
        surface_op={"op": "replace", "startSeq": 0, "endSeq": 1},
        source_event_seqs=[0, 1],
    )
    with pytest.raises(SurfaceError, match="end seq 1 not found in surface"):
        fold_surface([*events, replacement])


def test_derive_event_message_non_record_message() -> None:
    """Verifies a system/developer/assistant event without a message
    object projects nothing."""
    event = _event(0, "assistant/message", {"message": "not-an-object"})
    assert derive_event_message(event) is None
    tool_event = _event(0, "tool/result", {"message": "not-an-object"})
    assert derive_event_message(tool_event) is None


def test_fold_tool_result_rewrite_list_vs_scalar() -> None:
    """Verifies the content-only comparison distinguishes lists from
    scalars in nested message fields."""
    events = [_system(0), _user(1), _tool_result(2, text="3")]
    events[2].payload["message"]["stream"] = "x"
    replacement = _tool_result(3, text="4")
    replacement.payload["message"]["id"] = "t2"
    replacement.payload["message"]["stream"] = []
    replacement.surface_op = {"op": "replace", "startSeq": 2, "endSeq": 2}
    replacement.source_event_seqs = [2]
    with pytest.raises(SurfaceError, match="may change only content"):
        fold_surface([*events, replacement])


def test_surface_manager_generations_process_delta() -> None:
    """Verifies the generation properties catch the manager up on the
    log before reporting."""
    log = [_system(0), _user(1)]
    manager = SurfaceManager(log)
    replacement = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 1},
        source_event_seqs=[1],
    )
    log.append(replacement)
    assert manager.replace_generation == 1
    assert manager.content_generation == 1


def test_surface_manager_content_generation_processes_delta() -> None:
    """Verifies content_generation also catches the manager up on the
    log before reporting, when read first."""
    log = [_system(0), _user(1)]
    manager = SurfaceManager(log)
    replacement = _user(
        2,
        surface_op={"op": "replace", "startSeq": 1, "endSeq": 1},
        source_event_seqs=[1],
    )
    log.append(replacement)
    assert manager.content_generation == 1
