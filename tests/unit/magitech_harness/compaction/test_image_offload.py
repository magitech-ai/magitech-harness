"""Tests for the image-offload compaction seam."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from animas import Context

from magitech_harness.agent.cancellation import AbortSignal
from magitech_harness.compaction.image_offload import (
    IMAGE_OFFLOAD_PROJECTION,
    ImageOffloadProjection,
    _is_index,
    apply,
    offload_message_images,
    offload_oldest_images,
)
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.surface import MessageProjectionContext


def _event(seq: int, event_type: str, payload: dict[str, Any]) -> SessionEvent:
    return SessionEvent(
        id=f"e{seq}",
        session_id="s1",
        type=event_type,
        payload=payload,
        seq=seq,
    )


def _image(ref: str = "img", offloaded: bool = False) -> dict[str, Any]:
    return {
        "type": "image",
        "attachment": {"ref": ref},
        **({"offloaded": True} if offloaded else {}),
    }


def _text(text: str = "x") -> dict[str, Any]:
    return {"type": "text", "text": text}


# -- pure projection -------------------------------------------------------


def test_offload_message_images() -> None:
    message = {"role": "user", "content": [_text(), _image("a"), _text(), _image("b")]}
    result = offload_message_images(message, [1])
    assert result["role"] == "user"
    assert result["content"][3]["offloaded"] is True
    assert "offloaded" not in result["content"][1]
    assert message["content"][3] == _image("b")

    both = offload_message_images(message, [0, 1])
    assert both["content"][1]["offloaded"] is True
    assert both["content"][3]["offloaded"] is True

    with pytest.raises((ValueError, TypeError), match="already offloaded"):
        offload_message_images({"content": [_image(offloaded=True)]}, [0])
    with pytest.raises((ValueError, TypeError), match="does not exist"):
        offload_message_images({"content": [_text()]}, [0])
    with pytest.raises((ValueError, TypeError), match="content must be an array"):
        offload_message_images({}, [0])


def test_is_index() -> None:
    assert _is_index(0)
    assert _is_index(12)
    assert not _is_index(-1)
    assert not _is_index(True)
    assert not _is_index("1")


def test_projection_validation() -> None:
    projection = ImageOffloadProjection()
    good_event = _event(
        1, "image/offload", {"targets": [{"seq": 0, "imageIndexes": [0]}]}
    )
    source = _event(0, "user/message", {"role": "user", "content": [_image()]})
    context = MessageProjectionContext(
        nodes=[0], events=[source], base_seq=0, messages={}
    )
    result = projection.project(good_event, context)
    assert result[0]["content"][0]["offloaded"] is True

    bad_payloads = [
        {},
        {"targets": []},
        {"targets": [{}], "extra": 1},
        {"targets": [{"seq": 0}]},
        {"targets": [{"seq": 0, "imageIndexes": []}]},
        {"targets": [{"seq": True, "imageIndexes": [0]}]},
        {"targets": [{"seq": 5, "imageIndexes": [0]}]},
        {"targets": [{"seq": 0, "imageIndexes": [0]}, {"seq": 0, "imageIndexes": [1]}]},
    ]
    for payload in bad_payloads:
        with pytest.raises((ValueError, TypeError)):
            projection.project(_event(1, "image/offload", payload), context)

    with pytest.raises((ValueError, TypeError), match="outside the projection window"):
        projection.project(
            _event(1, "image/offload", {"targets": [{"seq": 9, "imageIndexes": [0]}]}),
            MessageProjectionContext(
                nodes=[9], events=[source], base_seq=0, messages={}
            ),
        )
    with pytest.raises((ValueError, TypeError), match="must be user/message or tool/result"):
        projection.project(
            _event(1, "image/offload", {"targets": [{"seq": 0, "imageIndexes": [0]}]}),
            MessageProjectionContext(
                nodes=[0], events=[_event(0, "other", {})], base_seq=0, messages={}
            ),
        )
    with pytest.raises((ValueError, TypeError), match="strictly increasing"):
        projection.project(
            _event(
                1, "image/offload", {"targets": [{"seq": 0, "imageIndexes": [0, 0]}]}
            ),
            context,
        )
    with pytest.raises((ValueError, TypeError), match="has no message"):
        projection.project(
            _event(1, "image/offload", {"targets": [{"seq": 0, "imageIndexes": [0]}]}),
            MessageProjectionContext(
                nodes=[0],
                events=[_event(0, "tool/result", {})],
                base_seq=0,
                messages={},
            ),
        )


def test_projection_uses_existing_and_tool_messages() -> None:
    projection = ImageOffloadProjection()
    source = _event(
        0, "tool/result", {"message": {"role": "tool", "content": [_image()]}}
    )
    context = MessageProjectionContext(
        nodes=[0], events=[source], base_seq=0, messages={}
    )
    event = _event(1, "image/offload", {"targets": [{"seq": 0, "imageIndexes": [0]}]})
    assert projection.project(event, context)[0]["content"][0]["offloaded"] is True

    existing = {"role": "user", "content": [_image()]}
    context2 = MessageProjectionContext(
        nodes=[0], events=[source], base_seq=0, messages={0: existing}
    )
    result = projection.project(event, context2)
    assert result[0]["content"][0]["offloaded"] is True
    assert "offloaded" not in existing["content"][0]
    assert IMAGE_OFFLOAD_PROJECTION.type == "image/offload"


# -- selection -------------------------------------------------------------


def test_offload_oldest_images() -> None:
    session = SimpleNamespace(
        event_at=lambda seq: (
            _event(seq, "user/message", {"content": [_text(), _image()]})
            if seq == 0
            else None
        ),
        append=lambda event_type, payload: None,
    )
    assert offload_oldest_images(session, [0], 1) is True
    assert offload_oldest_images(session, [0], 0) is False
    assert offload_oldest_images(session, [5], 1) is False

    already = SimpleNamespace(
        event_at=lambda seq: _event(
            seq, "user/message", {"content": [_image(offloaded=True)]}
        ),
        append=lambda event_type, payload: None,
    )
    assert offload_oldest_images(already, [0], 1) is False

    appended: list[Any] = []
    session2 = SimpleNamespace(
        event_at=lambda seq: _event(seq, "user/message", {"content": [_image()]}),
        append=lambda event_type, payload: appended.append((event_type, payload)),
    )
    assert offload_oldest_images(session2, [0], 1) is True
    assert appended[0][0] == "image/offload"
    assert appended[0][1]["targets"][0]["imageIndexes"] == [0]

    tool_session = SimpleNamespace(
        event_at=lambda seq: _event(
            seq, "tool/result", {"message": {"content": [_image()]}}
        ),
        append=lambda event_type, payload: None,
    )
    assert offload_oldest_images(tool_session, [0], 1) is True

    bad = SimpleNamespace(
        event_at=lambda seq: _event(seq, "user/message", {"content": "no"}),
        append=lambda event_type, payload: None,
    )
    assert offload_oldest_images(bad, [0], 1) is False


# -- wiring ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_and_listener() -> None:
    registered: list[Any] = []
    store = SimpleNamespace(
        register_message_projection=lambda projection: registered.append(projection)
    )
    ctx = Context()
    ctx.provide("sessions", SimpleNamespace(store=store))
    apply(ctx)
    await ctx.flush()
    assert registered == [IMAGE_OFFLOAD_PROJECTION]

    listener = ctx.events._listeners["agent/request-error"][0].callback
    session = SimpleNamespace(
        surface_nodes=lambda: [0],
        event_at=lambda seq: _event(seq, "user/message", {"content": [_image()]}),
        append=lambda event_type, payload: None,
    )
    calls: list[str] = []

    def next_fn() -> str:
        calls.append("next")
        return "next"

    def run(failure: Any, with_session: bool = True, aborted: bool = False) -> Any:
        signal = AbortSignal()
        if aborted:
            signal.abort()
        return listener(
            {
                "failure": failure,
                "session": session if with_session else None,
                "signal": signal,
            },
            next_fn,
        )

    assert run(None) == "next"
    assert run({"code": "OTHER"}) == "next"
    assert run({"code": "IMAGE_OFFLOAD_REQUIRED", "offloadImages": 0}) == "next"
    assert run({"code": "IMAGE_OFFLOAD_REQUIRED", "offloadImages": 1}, False) == "next"
    from magitech_harness.agent.cancellation import AbortError

    with pytest.raises(AbortError):
        run({"code": "IMAGE_OFFLOAD_REQUIRED", "offloadImages": 1}, True, True)
    assert run({"code": "IMAGE_OFFLOAD_REQUIRED", "offloadImages": 1}) == {
        "kind": "retry"
    }
    empty_session = SimpleNamespace(
        surface_nodes=list, event_at=lambda seq: None, append=lambda *a: None
    )
    assert (
        listener(
            {
                "failure": {"code": "IMAGE_OFFLOAD_REQUIRED", "offloadImages": 1},
                "session": empty_session,
            },
            next_fn,
        )
        == "next"
    )


def test_offload_selection_edge_branches() -> None:
    bad_message = SimpleNamespace(
        event_at=lambda seq: _event(seq, "tool/result", {"message": 123}),
        append=lambda *a: None,
    )
    assert offload_oldest_images(bad_message, [0], 1) is False

    multiple = SimpleNamespace(
        event_at=lambda seq: _event(
            seq, "user/message", {"content": [_image("a"), _image("b"), _image("c")]}
        ),
        append=lambda *a: None,
    )
    assert offload_oldest_images(multiple, [0], 1) is True
