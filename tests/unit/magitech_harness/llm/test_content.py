"""Unit tests for content block handles, image offloading, and tool-update projection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from magitech_harness.llm.brand import AttachmentId
from magitech_harness.llm.content import (
    content_has_file,
    content_has_image,
    file_handle_text,
    offloaded_image_text,
    project_files_to_text,
    project_images_for_text_model,
    project_offloaded_images,
    project_tool_updates,
    request_image_handle_text,
    required_image_offload,
    resolve_image_attachment_access,
    text_only_image_text,
)
from magitech_harness.llm.types import (
    FileAttachmentRef,
    ImageAttachmentRef,
    RequestMessage,
    ToolHistory,
    ToolSchema,
)


def _img(
    media_type: str = "image/png",
    name: str | None = "photo.png",
    bytes_count: int = 128,
) -> ImageAttachmentRef:
    ref: dict[str, Any] = {
        "kind": "image",
        "attachmentId": AttachmentId("sha256:abcdef0123456789"),
        "mediaType": media_type,
        "bytes": bytes_count,
        "width": 800,
        "height": 600,
        "sha256": "abcdef0123456789",
    }
    if name is not None:
        ref["name"] = name
    return ref  # type: ignore[return-value]


def _file(name: str = "notes.txt") -> FileAttachmentRef:
    return {
        "kind": "file",
        "attachmentId": AttachmentId("sha256:12345678abcdef"),
        "name": name,
        "mediaType": "text/plain",
        "bytes": 42,
        "sha256": "12345678abcdef",
    }


def test_resolve_image_attachment_access_and_handle_formatting() -> None:
    ref = _img()
    assert resolve_image_attachment_access(None, lambda p: p, ref) is None
    att_none = SimpleNamespace(imageHostPath=lambda _r: None)
    assert resolve_image_attachment_access(att_none, lambda p: p, ref) is None
    att_ok = SimpleNamespace(image_host_path=lambda _r: "/host/img.png")
    assert resolve_image_attachment_access(att_ok, lambda _p: None, ref) is None
    access = resolve_image_attachment_access(att_ok, lambda p: f"/proc{p}", ref)
    assert access == {"readonlyPath": "/proc/host/img.png"}

    for media_type, ext in (
        ("image/png", ".png"),
        ("image/jpeg", ".jpg"),
        ("image/webp", ".webp"),
        ("image/gif", ".gif"),
    ):
        r = _img(media_type=media_type, name=None)
        txt = request_image_handle_text(
            r,
            {
                "attachmentId": r["attachmentId"],
                "mediaType": media_type,
                "bytes": 10,
                "width": 100,
                "height": 100,
                "data": "AA==",
            },
            access,
        )
        assert ext in txt

    with pytest.raises(ValueError, match="unreachable variant in image extension"):
        request_image_handle_text(
            _img(media_type="image/bmp"),  # type: ignore[arg-type]
            {"width": 10, "height": 10},
            access,
        )

    assert "request preview 400x300px" in request_image_handle_text(
        ref, {"width": 400, "height": 300}, None
    )
    assert text_only_image_text(ref) == (
        "[image omitted because this model accepts text only; attachment sha256:abcdef01]"
    )
    assert "No local normalized image path is available" in offloaded_image_text(
        ref, None
    )
    assert "/proc/host/img.png" in offloaded_image_text(ref, access)


def test_file_and_image_projections_and_offload_budgets() -> None:
    fref = _file()
    iref = _img()
    msg_text: RequestMessage = {
        "id": "m1",
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
        "source": {"kind": "user"},
    }
    msg_file: RequestMessage = {
        "id": "m2",
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "file", "attachment": fref},
        ],
        "source": {"kind": "user"},
    }
    msg_img: RequestMessage = {
        "id": "m3",
        "role": "user",
        "content": [
            {"type": "image", "attachment": iref},
            {"type": "image", "attachment": iref, "offloaded": True},
            {"type": "text", "text": "tail"},
        ],
        "source": {"kind": "user"},
    }

    assert content_has_file(msg_file["content"])
    assert not content_has_file(msg_text["content"])
    assert content_has_image(msg_img["content"])
    assert not content_has_image(msg_text["content"])

    assert "cannot access a readable path" in file_handle_text(fref, None)
    assert "/proc/notes.txt" in file_handle_text(fref, "/proc/notes.txt")

    # No files/images -> returns exact original sequence by identity.
    assert project_files_to_text([msg_text], lambda _r: None) == [msg_text]
    msgs_with_file = [msg_text, msg_file]
    projected_files = project_files_to_text(msgs_with_file, lambda _r: "/tmp/notes.txt")
    assert projected_files[0] is msg_text
    assert projected_files[1]["content"][0] == {"type": "text", "text": "look"}
    assert "/tmp/notes.txt" in projected_files[1]["content"][1]["text"]

    # Offloaded images placeholder projection.
    assert project_offloaded_images(
        [msg_text], lambda r: f"off:{r['attachmentId']}"
    ) == [msg_text]
    proj_off = project_offloaded_images(
        [msg_text, msg_img], lambda r: f"off:{r['attachmentId']}"
    )
    assert proj_off[0] is msg_text
    assert proj_off[1]["content"][0]["type"] == "image"
    assert proj_off[1]["content"][1] == {
        "type": "text",
        "text": "off:sha256:abcdef0123456789",
    }
    assert proj_off[1]["content"][2] == {"type": "text", "text": "tail"}

    # Images replaced for text-only routes.
    assert project_images_for_text_model([msg_text]) == [msg_text]
    proj_text_model = project_images_for_text_model([msg_text, msg_img])
    assert proj_text_model[0] is msg_text
    assert proj_text_model[1]["content"][0]["text"].startswith(
        "[image omitted because this model accepts text only"
    )
    assert proj_text_model[1]["content"][2] == {"type": "text", "text": "tail"}

    # Required image offload budgets (bytes and base64 with/without quantum > 1).
    two_img_msgs: list[RequestMessage] = [
        {
            "id": "1",
            "role": "user",
            "content": [
                {"type": "image", "attachment": _img(bytes_count=30)},
                {"type": "image", "attachment": _img(bytes_count=30)},
                {
                    "type": "image",
                    "attachment": _img(bytes_count=30),
                    "offloaded": True,
                },
            ],
            "source": {"kind": "user"},
        },
    ]
    assert (
        required_image_offload(
            two_img_msgs,
            {"representation": "bytes", "maxImages": 5, "maxBytes": 200},
            lambda b: b["attachment"]["bytes"],
        )
        == 0
    )
    assert (
        required_image_offload(
            two_img_msgs,
            {"representation": "bytes", "maxImages": 1},
            lambda b: b["attachment"]["bytes"],
        )
        == 1
    )
    assert (
        required_image_offload(
            two_img_msgs,
            {"representation": "bytes", "maxBytes": 30},
            lambda b: b["attachment"]["bytes"],
        )
        == 1
    )
    assert (
        required_image_offload(
            two_img_msgs,
            {
                "representation": "base64",
                "maxImages": 1,
                "maxBytes": 40,
                "countQuantum": 2,
                "byteQuantum": 4,
            },
            lambda b: b["attachment"]["bytes"],
        )
        == 2
    )


def test_project_tool_updates_across_modes() -> None:
    t_alpha: ToolSchema = {
        "name": "alpha",
        "description": "A",
        "parameters": {},
        "deferLoading": True,
    }
    t_beta: ToolSchema = {"name": "beta", "description": "B", "parameters": {}}
    t_gamma: ToolSchema = {"name": "gamma", "description": "C", "parameters": {}}

    sys_msg: RequestMessage = {
        "id": "s",
        "role": "system",
        "content": [{"type": "text", "text": "sys"}],
        "source": {"kind": "system-prompt"},
    }
    dev_add: RequestMessage = {
        "id": "d1",
        "role": "developer",
        "content": [
            {"type": "tool-addition", "toolName": "alpha"},
            {"type": "tool-addition", "toolName": "alpha"},
            {"type": "text", "text": "keep"},
        ],
        "source": {"kind": "test"},
    }
    dev_rem: RequestMessage = {
        "id": "d2",
        "role": "developer",
        "content": [{"type": "tool-removal", "toolName": "beta"}],
        "source": {"kind": "test"},
    }
    dev_untracked: RequestMessage = {
        "id": "d3",
        "role": "developer",
        "content": [{"type": "text", "text": "untracked"}],
        "source": {"kind": "test"},
    }

    # Mode None strips developer messages and deferLoading.
    mode_none = project_tool_updates([sys_msg, dev_add], [t_alpha, t_beta], None)
    assert mode_none["messages"] == [sys_msg]
    assert mode_none["tools"] == [
        {"name": "alpha", "description": "A", "parameters": {}},
        t_beta,
    ]
    assert project_tool_updates([sys_msg], [t_beta], None)["tools"] == [t_beta]
    assert project_tool_updates([sys_msg], None, None)["tools"] is None

    # Missing history or missing developer message id drops developer messages.
    assert project_tool_updates([sys_msg, dev_add], [t_beta], "in-history", None)[
        "messages"
    ] == [sys_msg]
    history_missing: ToolHistory = {
        "tools": [t_beta],
        "updates": [{"messageId": "missing", "additions": []}],
    }
    assert project_tool_updates(
        [sys_msg, dev_add], [t_beta], "in-history", history_missing
    )["messages"] == [sys_msg]

    # in-history keeps valid additions + removals and drops duplicate/untracked blocks.
    history_full: ToolHistory = {
        "tools": [
            t_beta,
            {
                "name": "gamma",
                "description": "C",
                "parameters": {},
                "deferLoading": True,
            },
        ],
        "updates": [
            {"messageId": "d1", "additions": [t_alpha]},
            {"messageId": "d2", "additions": []},
        ],
    }
    res_in_hist = project_tool_updates(
        [sys_msg, dev_add, dev_rem, dev_untracked], [t_beta], "in-history", history_full
    )
    assert [m["id"] for m in res_in_hist["messages"]] == ["s", "d1", "d2"]
    assert res_in_hist["messages"][1]["content"] == [
        {"type": "tool-addition", "toolName": "alpha"},
        {"type": "text", "text": "keep"},
    ]
    assert any(
        t["name"] == "alpha" and t.get("deferLoading") is True
        for t in res_in_hist["tools"]
    )

    # addition-only rebases baseline on removal and drops removal messages cleanly.
    history_rebase: ToolHistory = {
        "tools": [t_beta, t_gamma],
        "updates": [
            {"messageId": "d2", "additions": []},
            {"messageId": "d1", "additions": [t_alpha]},
        ],
    }
    res_add_only = project_tool_updates(
        [sys_msg, dev_rem, dev_add], [t_beta], "addition-only", history_rebase
    )
    assert [m["id"] for m in res_add_only["messages"]] == ["s", "d1"]
    assert [t["name"] for t in res_add_only["tools"]] == ["gamma", "alpha"]

    # addition-only when reset=True due to empty baseline -> additions.
    dev_add_only: RequestMessage = {
        "id": "d_empty_base",
        "role": "developer",
        "content": [{"type": "tool-addition", "toolName": "alpha"}],
        "source": {"kind": "test"},
    }
    res_empty_base = project_tool_updates(
        [sys_msg, dev_add_only],
        [],
        "addition-only",
        {
            "tools": [],
            "updates": [{"messageId": "d_empty_base", "additions": [t_alpha]}],
        },
    )
    assert res_empty_base["messages"] == [sys_msg]
    assert res_empty_base["tools"] == [
        {"name": "alpha", "description": "A", "parameters": {}}
    ]

    # Cover unchanged=True branch when addition-only returns exact original messages & declarations already hold name.
    res_unchanged = project_tool_updates(
        [sys_msg],
        [t_beta],
        "addition-only",
        {"tools": [t_beta], "updates": []},
    )
    assert res_unchanged["messages"] == [sys_msg]

    # Pre-offered tool in baseline + addition update deduplicates cleanly; removal of non-offered tool skipped.
    dev_dup_rem: RequestMessage = {
        "id": "d4",
        "role": "developer",
        "content": [
            {"type": "tool-addition", "toolName": "beta"},
            {"type": "tool-removal", "toolName": "missing"},
        ],
        "source": {"kind": "test"},
    }
    res_skip = project_tool_updates(
        [sys_msg, dev_dup_rem],
        [t_beta],
        "in-history",
        {"tools": [t_beta], "updates": [{"messageId": "d4", "additions": [t_beta]}]},
    )
    assert res_skip["messages"] == [sys_msg]

    # Multiple file blocks and offloaded images interleaved with text blocks.
    multi_msg: RequestMessage = {
        "id": "u-multi",
        "role": "user",
        "content": [
            {"type": "file", "attachment": _file()},
            {"type": "file", "attachment": _file(name="second.txt")},
            {"type": "image", "attachment": _img(), "offloaded": True},
            {"type": "image", "attachment": _img(), "offloaded": True},
            {"type": "text", "text": "tail"},
        ],
        "source": {"kind": "test"},
    }
    projected_f = project_files_to_text([multi_msg], lambda _ref: "/p/a")
    assert projected_f[0]["content"][0]["type"] == "text"
    assert projected_f[0]["content"][1]["type"] == "text"
    assert projected_f[0]["content"][4] == {"type": "text", "text": "tail"}
    projected_o = project_offloaded_images([multi_msg], lambda _ref: "/p/i.png")
    assert projected_o[0]["content"][2]["type"] == "text"
    assert projected_o[0]["content"][3]["type"] == "text"
    assert projected_o[0]["content"][4] == {"type": "text", "text": "tail"}
