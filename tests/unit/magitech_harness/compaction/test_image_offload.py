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
    with pytest.raises(
        (ValueError, TypeError), match="must be user/message or tool/result"
    ):
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


def _derive(event: SessionEvent) -> Any:
    payload = event.payload
    return payload if event.type == "user/message" else payload.get("message")


def _stub_session(
    events: dict[int, SessionEvent], appended: list[Any] | None = None
) -> Any:
    return SimpleNamespace(
        event_at=lambda seq: events.get(seq),
        derive_event_message=_derive,
        surface_nodes=lambda: list(events),
        append=lambda event_type, payload: (
            appended.append((event_type, payload)) if appended is not None else None
        ),
    )


def test_offload_oldest_images() -> None:
    events = {0: _event(0, "user/message", {"content": [_text(), _image()]})}
    session = _stub_session(events)
    assert offload_oldest_images(session, [0], 1) is True
    assert offload_oldest_images(session, [0], 0) is False
    assert offload_oldest_images(session, [5], 1) is False

    already = _stub_session(
        {0: _event(0, "user/message", {"content": [_image(offloaded=True)]})}
    )
    assert offload_oldest_images(already, [0], 1) is False

    appended: list[Any] = []
    session2 = _stub_session(
        {0: _event(0, "user/message", {"content": [_image()]})}, appended
    )
    assert offload_oldest_images(session2, [0], 1) is True
    assert appended[0][0] == "image/offload"
    assert appended[0][1]["targets"][0]["imageIndexes"] == [0]

    tool_session = _stub_session(
        {0: _event(0, "tool/result", {"message": {"content": [_image()]}})}
    )
    assert offload_oldest_images(tool_session, [0], 1) is True

    bad = _stub_session({0: _event(0, "user/message", {"content": "no"})})
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

    session = _stub_session({0: _event(0, "user/message", {"content": [_image()]})})
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
    assert (
        listener(
            {
                "failure": {"code": "IMAGE_OFFLOAD_REQUIRED", "offloadImages": 1},
                "session": _stub_session({}),
            },
            next_fn,
        )
        == "next"
    )


def test_offload_selection_edge_branches() -> None:
    bad_message = _stub_session({0: _event(0, "tool/result", {"message": 123})})
    assert offload_oldest_images(bad_message, [0], 1) is False

    multiple = _stub_session(
        {
            0: _event(
                0, "user/message", {"content": [_image("a"), _image("b"), _image("c")]}
            )
        }
    )
    assert offload_oldest_images(multiple, [0], 1) is True


def test_consecutive_failures_advance_indexes() -> None:
    """A second offload must select the next retained image, not repeat index 0."""
    from magitech_harness.session.store import Session

    session = Session(
        "offload-sequence",
        projections=[IMAGE_OFFLOAD_PROJECTION],
    )
    session.append(
        "user/message",
        {
            "role": "user",
            "content": [_text("a"), _image("one"), _image("two")],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )
    first = offload_oldest_images(session, session.surface_nodes(), 1)
    assert first is True
    assert session.events[-1].payload["targets"][0]["imageIndexes"] == [0]

    second = offload_oldest_images(session, session.surface_nodes(), 1)
    assert second is True
    assert session.events[-1].payload["targets"][0]["imageIndexes"] == [1]

    third = offload_oldest_images(session, session.surface_nodes(), 1)
    assert third is False
