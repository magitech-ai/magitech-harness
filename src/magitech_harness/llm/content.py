"""Content-block structure, attachment handle formatting, and route projections."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict, cast

from magitech_harness.llm.types import (
    ContentBlock,
    FileAttachmentRef,
    ImageAttachmentRef,
    ImageBlock,
    ImageMediaType,
    RequestImageAttachment,
    RequestMessage,
    ToolHistory,
    ToolSchema,
    ToolUpdate,
)
from magitech_harness.llm.values import assert_never


class ImageAttachmentAccess(TypedDict):
    """Execution-world read-only path resolved for an image attachment."""

    readonlyPath: str


ImageAttachmentAccessResolver = Callable[
    [ImageAttachmentRef], ImageAttachmentAccess | None
]


def resolve_image_attachment_access(
    attachments: Any,
    map_host_path: Callable[[str], str | None],
    ref: ImageAttachmentRef,
) -> ImageAttachmentAccess | None:
    """Resolve current execution-world read-only path for a normalized image ref."""
    resolver = getattr(attachments, "imageHostPath", None) or getattr(
        attachments, "image_host_path", None
    )
    if not callable(resolver):
        return None
    host_path = resolver(ref)
    if host_path is None:
        return None
    readonly_path = map_host_path(host_path)
    return None if readonly_path is None else {"readonlyPath": readonly_path}


def _image_identity(ref: ImageAttachmentRef) -> str:
    name = ref.get("name")
    if name is None:
        return str(ref["attachmentId"])
    return f"{json.dumps(name)} ({ref['attachmentId']})"


def _image_extension(media_type: ImageMediaType) -> str:
    if media_type == "image/png":
        return ".png"
    if media_type == "image/jpeg":
        return ".jpg"
    if media_type == "image/webp":
        return ".webp"
    if media_type == "image/gif":
        return ".gif"
    assert_never(media_type, "image extension")


def _normalized_access_text(
    ref: ImageAttachmentRef, access: ImageAttachmentAccess
) -> str:
    return (
        " Normalized copy (read-only; may be resized or re-encoded): "
        f"{json.dumps(access['readonlyPath'])} "
        f"({ref['width']}x{ref['height']}px, {ref['mediaType']})."
        " Source dimensions, format, and byte size may differ."
        f" Copy to a writable path ending in {_image_extension(ref['mediaType'])}"
        " before editing."
    )


def text_only_image_text(ref: ImageAttachmentRef) -> str:
    """Build deterministic text placeholder for text-only model routes."""
    prefix = "sha256:"
    digest = str(ref["attachmentId"])[len(prefix) : len(prefix) + 8]
    return (
        "[image omitted because this model accepts text only; "
        f"attachment sha256:{digest}]"
    )


def request_image_handle_text(
    ref: ImageAttachmentRef,
    version: RequestImageAttachment | Mapping[str, Any],
    access: ImageAttachmentAccess | None = None,
) -> str:
    """Build model-facing handle text for an exact request preview image."""
    preview = (
        f"Image {_image_identity(ref)}; "
        f"request preview {version['width']}x{version['height']}px."
    )
    if access is None:
        return (
            f"{preview} It may be resized or re-encoded; "
            "source dimensions, format, and byte size may differ."
        )
    return preview + _normalized_access_text(ref, access)


def offloaded_image_text(
    ref: ImageAttachmentRef, access: ImageAttachmentAccess | None = None
) -> str:
    """Build deterministic placeholder text for an image omitted to fit request limits."""
    identity = f"image omitted to fit request image limits; {_image_identity(ref)}."
    if access is None:
        return (
            f"[{identity} No local normalized image path is available; "
            "ask the user to attach it again if needed.]"
        )
    return f"[{identity}{_normalized_access_text(ref, access)}]"


def content_has_image(content: Sequence[ContentBlock]) -> bool:
    """Return True if any content block has type 'image'."""
    return any(block.get("type") == "image" for block in content)


def content_has_file(content: Sequence[ContentBlock]) -> bool:
    """Return True if any content block has type 'file'."""
    return any(block.get("type") == "file" for block in content)


def file_handle_text(ref: FileAttachmentRef, readonly_path: str | None) -> str:
    """Build deterministic model-facing handle text for a durable file attachment."""
    prefix = "sha256:"
    digest = str(ref["attachmentId"])[len(prefix) : len(prefix) + 8]
    identity = f"File {json.dumps(ref['name'])} ({ref['bytes']} bytes, sha256:{digest})"
    if readonly_path is None:
        return (
            f"[{identity} was uploaded, but the current execution environment "
            "cannot access a readable path. Report that limitation if its contents "
            "are needed; do not claim to have read it.]"
        )
    return (
        f"[{identity}: verbatim read-only copy saved at {json.dumps(readonly_path)}. "
        "Read that path with your file tools when its contents are needed; copy it to "
        "a writable location before modifying it. When delegating file work, include "
        "this saved path in the delegation prompt; only subagents sharing this "
        "execution environment can read it.]"
    )


def _replace_files_with_handles(
    blocks: Sequence[ContentBlock],
    resolve_path: Callable[[FileAttachmentRef], str | None],
) -> Sequence[ContentBlock]:
    rewritten: list[ContentBlock] | None = None
    for index, block in enumerate(blocks):
        if block.get("type") == "file":
            if rewritten is None:
                rewritten = list(blocks[:index])
            attachment = cast(
                FileAttachmentRef, cast(Mapping[str, Any], block)["attachment"]
            )
            rewritten.append(
                {
                    "type": "text",
                    "text": file_handle_text(attachment, resolve_path(attachment)),
                }
            )
            continue
        if rewritten is not None:
            rewritten.append(block)
    return blocks if rewritten is None else rewritten


def project_files_to_text(
    messages: Sequence[RequestMessage],
    resolve_path: Callable[[FileAttachmentRef], str | None],
) -> Sequence[RequestMessage]:
    """Project all file blocks in messages to handle text blocks."""
    if not any(content_has_file(msg["content"]) for msg in messages):
        return messages
    projected: list[RequestMessage] = []
    for msg in messages:
        new_content = _replace_files_with_handles(msg["content"], resolve_path)
        projected.append(
            msg if new_content is msg["content"] else {**msg, "content": new_content}
        )
    return projected


def _replace_offloaded_images(
    blocks: Sequence[ContentBlock],
    placeholder: Callable[[ImageAttachmentRef], str],
) -> Sequence[ContentBlock]:
    rewritten: list[ContentBlock] | None = None
    for index, block in enumerate(blocks):
        if block.get("type") == "image" and block.get("offloaded") is True:
            if rewritten is None:
                rewritten = list(blocks[:index])
            attachment = cast(
                ImageAttachmentRef, cast(Mapping[str, Any], block)["attachment"]
            )
            rewritten.append({"type": "text", "text": placeholder(attachment)})
            continue
        if rewritten is not None:
            rewritten.append(block)
    return blocks if rewritten is None else rewritten


def project_offloaded_images(
    messages: Sequence[RequestMessage],
    placeholder: Callable[[ImageAttachmentRef], str],
) -> Sequence[RequestMessage]:
    """Replace offloaded image blocks with caller-provided placeholder text."""
    changed = False
    projected: list[RequestMessage] = []
    for msg in messages:
        new_content = _replace_offloaded_images(msg["content"], placeholder)
        if new_content is msg["content"]:
            projected.append(msg)
        else:
            changed = True
            projected.append({**msg, "content": new_content})
    return projected if changed else messages


def _base64_length(byte_count: int) -> int:
    return math.ceil(byte_count / 3) * 4


def _offloaded_image_prefix_count(
    lengths: Sequence[int], budget: Mapping[str, Any]
) -> int:
    total = sum(lengths)
    max_images = budget.get("maxImages")
    max_bytes = budget.get("maxBytes")
    excess_count = 0 if max_images is None else max(0, len(lengths) - max_images)
    excess_bytes = 0 if max_bytes is None else max(0, total - max_bytes)
    if excess_count == 0 and excess_bytes == 0:
        return 0
    raw_cq = budget.get("countQuantum")
    raw_bq = budget.get("byteQuantum")
    count_quantum = 1 if raw_cq is None else raw_cq
    byte_quantum = 1 if raw_bq is None else raw_bq
    remove_count = (
        0
        if excess_count == 0
        else math.ceil(excess_count / count_quantum) * count_quantum
    )
    remove_bytes = (
        0
        if excess_bytes == 0
        else math.ceil(excess_bytes / byte_quantum) * byte_quantum
    )
    count = 0
    removed_bytes = 0
    for image_bytes in lengths:
        byte_target_met = remove_bytes == 0 or (
            removed_bytes >= remove_bytes
            if byte_quantum == 1
            else removed_bytes > remove_bytes
        )
        if count >= remove_count and byte_target_met:
            break
        removed_bytes += image_bytes
        count += 1
    return count


def required_image_offload(
    messages: Sequence[RequestMessage],
    budget: Mapping[str, Any],
    version_bytes: Callable[[ImageBlock], int],
) -> int:
    """Compute how many leading retained image occurrences must be offloaded."""
    representation = budget["representation"]
    lengths: list[int] = []
    for msg in messages:
        for block in msg["content"]:
            if block.get("type") != "image" or block.get("offloaded") is True:
                continue
            byte_len = version_bytes(block)  # type: ignore[arg-type]
            lengths.append(
                _base64_length(byte_len) if representation == "base64" else byte_len
            )
    return _offloaded_image_prefix_count(lengths, budget)


def _replace_images_for_text_model(
    blocks: Sequence[ContentBlock],
) -> Sequence[ContentBlock]:
    rewritten: list[ContentBlock] | None = None
    for index, block in enumerate(blocks):
        if block.get("type") == "image":
            if rewritten is None:
                rewritten = list(blocks[:index])
            attachment = cast(
                ImageAttachmentRef, cast(Mapping[str, Any], block)["attachment"]
            )
            rewritten.append({"type": "text", "text": text_only_image_text(attachment)})
            continue
        if rewritten is not None:
            rewritten.append(block)
    return blocks if rewritten is None else rewritten


def project_images_for_text_model(
    messages: Sequence[RequestMessage],
) -> Sequence[RequestMessage]:
    """Replace all image blocks with text-only placeholders for non-vision routes."""
    if not any(content_has_image(msg["content"]) for msg in messages):
        return messages
    projected: list[RequestMessage] = []
    for msg in messages:
        new_content = _replace_images_for_text_model(msg["content"])
        projected.append(
            msg if new_content is msg["content"] else {**msg, "content": new_content}
        )
    return projected


def _without_developer_messages(
    messages: Sequence[RequestMessage],
) -> Sequence[RequestMessage]:
    retained = [msg for msg in messages if msg.get("role") != "developer"]
    return messages if len(retained) == len(messages) else retained


def _tool_declarations(history: ToolHistory) -> dict[str, ToolSchema]:
    declarations: dict[str, ToolSchema] = {
        tool["name"]: tool for tool in history["tools"]
    }
    for update in history["updates"]:
        for tool in update["additions"]:
            if tool["name"] not in declarations:
                declarations[tool["name"]] = {**tool, "deferLoading": True}
    return declarations


def _rebase_addition_only_history(
    history: ToolHistory,
    messages: Mapping[str, Sequence[ContentBlock]],
) -> ToolHistory:
    available: dict[str, ToolSchema] = {tool["name"]: tool for tool in history["tools"]}
    baseline = history["tools"]
    retained_from = 0
    for index, update in enumerate(history["updates"]):
        additions = {tool["name"]: tool for tool in update["additions"]}
        reset = len(available) == 0 and len(additions) > 0
        for block in messages[update["messageId"]]:
            btype = block.get("type")
            if btype == "tool-addition":
                tool = additions[cast(str, cast(Mapping[str, Any], block)["toolName"])]
                without_defer = {k: v for k, v in tool.items() if k != "deferLoading"}
                available[tool["name"]] = without_defer  # type: ignore[assignment]
            elif btype == "tool-removal":
                reset = True
                available.pop(
                    cast(str, cast(Mapping[str, Any], block)["toolName"]), None
                )
        if reset:
            baseline = list(available.values())
            retained_from = index + 1
    if retained_from == 0:
        return history
    return {"tools": baseline, "updates": list(history["updates"][retained_from:])}


def project_tool_updates(
    messages: Sequence[RequestMessage],
    tools: Sequence[ToolSchema] | None,
    tool_update: ToolUpdate | None,
    history: ToolHistory | None = None,
) -> dict[str, Any]:
    """Construct route-specific tool declarations and filtered developer updates."""
    if tool_update is None:
        immediate_tools = tools
        if tools is not None and any(t.get("deferLoading") is True for t in tools):
            immediate_tools = [
                {k: v for k, v in tool.items() if k != "deferLoading"}  # type: ignore[misc]
                for tool in tools
            ]
        return {
            "messages": _without_developer_messages(messages),
            "tools": immediate_tools,
        }

    if history is None:
        return {"messages": _without_developer_messages(messages), "tools": tools}

    updates: dict[str, Sequence[ContentBlock]] = {
        msg["id"]: msg["content"] for msg in messages if msg.get("role") == "developer"
    }
    if any(update["messageId"] not in updates for update in history["updates"]):
        return {"messages": _without_developer_messages(messages), "tools": tools}

    projected_history = (
        _rebase_addition_only_history(history, updates)
        if tool_update == "addition-only"
        else history
    )
    declarations = _tool_declarations(projected_history)
    update_ids = {update["messageId"] for update in history["updates"]}
    retained_update_ids = {
        update["messageId"] for update in projected_history["updates"]
    }
    offered = {
        tool["name"]
        for tool in projected_history["tools"]
        if not tool.get("deferLoading")
    }
    projected_messages: list[RequestMessage] = []
    for msg in messages:
        if msg.get("role") != "developer":
            projected_messages.append(msg)
            continue
        msg_id = msg["id"]
        if msg_id not in update_ids:
            continue
        kept_blocks: list[ContentBlock] = []
        for block in msg["content"]:
            btype = block.get("type")
            if btype == "tool-addition":
                if msg_id not in retained_update_ids:
                    continue
                tname = block["toolName"]
                if tname not in declarations or tname in offered:
                    continue
                offered.add(tname)
                kept_blocks.append(block)
            elif btype == "tool-removal":
                if tool_update != "in-history":
                    continue
                tname = block["toolName"]
                if tname in offered:
                    offered.remove(tname)
                    kept_blocks.append(block)
            else:
                kept_blocks.append(block)
        if not kept_blocks:
            continue
        if len(kept_blocks) == len(msg["content"]):
            projected_messages.append(msg)
        else:
            projected_messages.append({**msg, "content": kept_blocks})

    unchanged = len(projected_messages) == len(messages) and all(
        projected is original
        for projected, original in zip(projected_messages, messages, strict=True)
    )
    return {
        "messages": messages if unchanged else projected_messages,
        "tools": list(declarations.values()),
    }


resolveImageAttachmentAccess = resolve_image_attachment_access
textOnlyImageText = text_only_image_text
requestImageHandleText = request_image_handle_text
offloadedImageText = offloaded_image_text
contentHasImage = content_has_image
contentHasFile = content_has_file
fileHandleText = file_handle_text
projectFilesToText = project_files_to_text
projectOffloadedImages = project_offloaded_images
requiredImageOffload = required_image_offload
projectImagesForTextModel = project_images_for_text_model
projectToolUpdates = project_tool_updates
