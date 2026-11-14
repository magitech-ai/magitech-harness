"""Message factories: one home for identity minting and source-kind rules.

Every conversation message is one identified value: a fresh id, a role,
content blocks, and a producer-owned source. The loop and the tests
build the same shapes through these factories.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any


def create_message(
    *,
    role: str,
    content: list[dict[str, Any]],
    source: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    """One identified message: fresh id, detached inputs."""
    return {
        "id": uuid.uuid4().hex,
        "role": role,
        "content": copy.deepcopy(content),
        "source": copy.deepcopy(source),
        **copy.deepcopy(extra),
    }


def create_system_message(text: str) -> dict[str, Any]:
    """A system message: empty content for an empty prompt, else one text block."""
    return create_message(
        role="system",
        content=[] if text == "" else [{"type": "text", "text": text}],
        source={"kind": "system-prompt"},
    )


def create_user_message(
    *,
    content: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    """A user message with the caller's content blocks and source."""
    return create_message(role="user", content=content, source=source)


def create_developer_message(
    *,
    content: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    """A developer message with the caller's content blocks and source."""
    return create_message(role="developer", content=content, source=source)


def create_assistant_message(
    *,
    content: list[dict[str, Any]],
    provider: str,
    model: str,
) -> dict[str, Any]:
    """An assistant message sourced from one provider/model pair."""
    return create_message(
        role="assistant",
        content=content,
        source={"kind": "model", "provider": provider, "model": model},
    )


def create_tool_result_message(
    *,
    call_id: str,
    content: list[dict[str, Any]],
    is_error: bool = False,
) -> dict[str, Any]:
    """A tool-role message whose id matches its tool source's call id."""
    return create_message(
        role="tool",
        content=content,
        source={"kind": "tool", "callId": call_id},
        toolCallId=call_id,
        isError=is_error,
    )
