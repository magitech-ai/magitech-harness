"""Image offload executor for the compaction seam.

Ports ``@deepseek-ai/dsh-compaction-image-offload`` from the pinned dsh source
(``packages/compaction/compaction-image-offload/src``). When an image-capable
route fails a request with ``IMAGE_OFFLOAD_REQUIRED``, a durable
``image/offload`` decision selects the oldest retained input occurrences and
the request is retried. Every later request renders those occurrences as
placeholder text through the existing ``ctx.llm`` projection.
"""

from __future__ import annotations

from typing import Any

from animas import Context, EventMode

from magitech_harness.llm.error import IMAGE_OFFLOAD_REQUIRED_CODE
from magitech_harness.session.events import SessionEvent
from magitech_harness.session.surface import (
    MessageProjection,
    MessageProjectionContext,
)


def offload_message_images(
    message: dict[str, Any], indexes: list[int]
) -> dict[str, Any]:
    """Mark selected image occurrences as offloaded without changing identity."""
    content = message.get("content")
    if not isinstance(content, list):
        raise TypeError("image/offload: message.content must be an array")
    image_index = 0
    selected = 0
    rewritten: list[Any] | None = None
    for position, block in enumerate(content):
        projected = block
        if isinstance(block, dict) and block.get("type") == "image":
            if selected < len(indexes) and image_index == indexes[selected]:
                if block.get("offloaded") is True:
                    raise ValueError(
                        f"image/offload: image index {image_index} is already offloaded"
                    )
                projected = {**block, "offloaded": True}
                selected += 1
            image_index += 1
        if projected is not block:
            if rewritten is None:
                rewritten = list(content[:position])
            rewritten.append(projected)
        elif rewritten is not None:
            rewritten.append(block)
    if selected != len(indexes):
        raise ValueError(
            f"image/offload: image index {indexes[selected] if selected < len(indexes) else '?'} does not exist"
        )
    return {**message, "content": content if rewritten is None else rewritten}


class ImageOffloadProjection:
    """Durable, browser-safe interpretation of one ``image/offload`` decision."""

    type = "image/offload"

    def project(
        self, event: SessionEvent, context: MessageProjectionContext
    ) -> dict[int, dict[str, Any]]:
        data = event.payload
        if (
            not isinstance(data, dict)
            or set(data) != {"targets"}
            or not isinstance(data["targets"], list)
            or not data["targets"]
        ):
            raise ValueError(
                "image/offload: data must contain a nonempty targets array"
            )
        nodes = set(context.nodes)
        messages: dict[int, dict[str, Any]] = {}
        for target in data["targets"]:
            if (
                not isinstance(target, dict)
                or set(target) != {"seq", "imageIndexes"}
                or not _is_index(target["seq"])
                or not isinstance(target["imageIndexes"], list)
                or not target["imageIndexes"]
            ):
                raise ValueError(
                    "image/offload: each target must contain a seq and nonempty imageIndexes"
                )
            seq = target["seq"]
            if seq in messages:
                raise ValueError(f"image/offload: duplicate target seq {seq}")
            if seq not in nodes:
                raise ValueError(
                    f"image/offload: target seq {seq} is not a current surface node"
                )
            offset = seq - context.base_seq
            if offset < 0 or offset >= len(context.events):
                raise ValueError(
                    f"image/offload: target seq {seq} is outside the projection window"
                )
            source = context.events[offset]
            if source.type not in ("user/message", "tool/result"):
                raise ValueError(
                    f"image/offload: target seq {seq} must be user/message or tool/result"
                )
            previous = -1
            for index in target["imageIndexes"]:
                if not _is_index(index) or index <= previous:
                    raise ValueError(
                        "image/offload: imageIndexes must be strictly increasing non-negative safe integers"
                    )
                previous = index
            message = context.messages.get(seq)
            if message is None:
                payload = source.payload
                message = (
                    payload if source.type == "user/message" else payload.get("message")
                )
            if not isinstance(message, dict):
                raise TypeError(f"image/offload: target seq {seq} has no message")
            messages[seq] = offload_message_images(message, target["imageIndexes"])
        return messages


def _is_index(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        and value <= 2**53 - 1
    )


def offload_oldest_images(
    session: Any, source_event_seqs: list[int], count: int
) -> bool:
    """Log permanent omission of the oldest retained input-image occurrences."""
    targets: list[dict[str, Any]] = []
    for seq in source_event_seqs:
        if count == 0:
            break
        event = session.event_at(seq)
        if event is None or event.type not in ("user/message", "tool/result"):
            continue
        # Read the PROJECTED message so earlier offloads are visible; raw
        # payloads never carry the derived ``offloaded`` marks.
        message = session.derive_event_message(event)
        if not isinstance(message, dict):
            continue
        image_indexes: list[int] = []
        image_index = 0
        for block in message.get("content", []):
            if count == 0:
                break
            if isinstance(block, dict) and block.get("type") == "image":
                if block.get("offloaded") is not True:
                    image_indexes.append(image_index)
                    count -= 1
                image_index += 1
        if image_indexes:
            targets.append({"seq": seq, "imageIndexes": image_indexes})
    if not targets:
        return False
    session.append("image/offload", {"targets": targets})
    return True


IMAGE_OFFLOAD_PROJECTION: MessageProjection = ImageOffloadProjection()


def apply(ctx: Context, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Register the projection and install the request-error recovery listener."""
    del config, kwargs
    dispose_projection = ctx.sessions.store.register_message_projection(
        IMAGE_OFFLOAD_PROJECTION
    )
    ctx.effect(
        lambda: dispose_projection, "compaction-image-offload: remove projection"
    )

    def _on_error(payload: dict[str, Any], next_fn: Any) -> Any:
        failure = payload.get("failure")
        if not isinstance(failure, dict):
            return next_fn()
        if failure.get("code") != IMAGE_OFFLOAD_REQUIRED_CODE:
            return next_fn()
        count = failure.get("offloadImages")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            return next_fn()
        session = payload.get("session")
        if session is None:
            agent = payload.get("agent")
            session = getattr(agent, "session", None)
        if session is None:
            return next_fn()
        signal = payload.get("signal")
        if signal is not None:
            signal.throw_if_aborted()
        if offload_oldest_images(session, session.surface_nodes(), count):
            return {"kind": "retry"}
        return next_fn()

    ctx.on("agent/request-error", _on_error, mode=EventMode.WATERFALL, prepend=True)


apply.inject = ["sessions"]  # type: ignore[attr-defined]

__all__ = [
    "IMAGE_OFFLOAD_PROJECTION",
    "IMAGE_OFFLOAD_REQUIRED_CODE",
    "ImageOffloadProjection",
    "apply",
    "offload_message_images",
    "offload_oldest_images",
]
