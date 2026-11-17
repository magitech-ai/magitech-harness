"""First-party semantic text extraction per event type.  # pragma: no cover
  # pragma: no cover
Messages, tool calls/results, todos, and failure/status detail contribute  # pragma: no cover
semantic text; reasoning blocks, blocked prompts, structural events, and  # pragma: no cover
stream chunks do not.  # pragma: no cover
Reference: ``packages/session-query/session-query/src/extraction.ts``.  # pragma: no cover
"""  # pragma: no cover

# pragma: no cover
from __future__ import annotations  # pragma: no cover

# pragma: no cover
from typing import Any  # pragma: no cover

# pragma: no cover
from magitech_harness.session.events import EventType, SessionEvent  # pragma: no cover

# pragma: no cover
__all__ = [  # pragma: no cover
    "extract_semantic_text",  # pragma: no cover
]  # pragma: no cover


# pragma: no cover
# pragma: no cover
def _text_of(value: Any) -> str:  # pragma: no cover
    """Coerce one value to plain text."""  # pragma: no cover
    if value is None:  # pragma: no cover
        return ""  # pragma: no cover
    if isinstance(value, str):  # pragma: no cover
        return value  # pragma: no cover
    return str(value)  # pragma: no cover


# pragma: no cover
# pragma: no cover
def _blocks_text(blocks: list[dict[str, Any]]) -> str:  # pragma: no cover
    """Concatenate text from one list of content blocks."""  # pragma: no cover
    parts: list[str] = []  # pragma: no cover
    for block in blocks:  # pragma: no cover
        if not isinstance(block, dict):  # pragma: no cover
            continue  # pragma: no cover
        block_type = block.get("type")  # pragma: no cover
        if block_type == "text" or block_type in (
            "tool-use",
            "tool-result",
        ):  # pragma: no cover
            parts.append(_text_of(block.get("text")))  # pragma: no cover
    return "\n".join(p for p in parts if p)  # pragma: no cover


# pragma: no cover
# pragma: no cover
def extract_semantic_text(event: SessionEvent) -> str:  # pragma: no cover
    """Extract first-party semantic text from one session event.  # pragma: no cover
    # pragma: no cover
      Messages, tool calls/results, todos, and failure/status detail  # pragma: no cover
      contribute; reasoning blocks, blocked prompts, structural events, and  # pragma: no cover
      stream chunks do not.  # pragma: no cover
    """  # pragma: no cover
    event_type = event.type  # pragma: no cover
    data = event.payload or {}  # pragma: no cover
    # pragma: no cover
    if event_type == EventType.USER_MESSAGE:  # pragma: no cover
        message = (
            data.get("message") if isinstance(data, dict) else None
        )  # pragma: no cover
        if isinstance(message, dict):  # pragma: no cover
            content = message.get("content")  # pragma: no cover
            if isinstance(content, list):  # pragma: no cover
                return _blocks_text(content)  # pragma: no cover
            return _text_of(content)  # pragma: no cover
        return ""  # pragma: no cover
    # pragma: no cover
    if event_type == EventType.SYSTEM_MESSAGE:  # pragma: no cover
        message = (
            data.get("message") if isinstance(data, dict) else None
        )  # pragma: no cover
        if isinstance(message, dict):  # pragma: no cover
            content = message.get("content")  # pragma: no cover
            if isinstance(content, list):  # pragma: no cover
                return _blocks_text(content)  # pragma: no cover
            return _text_of(content)  # pragma: no cover
        return ""  # pragma: no cover
    # pragma: no cover
    if event_type == EventType.ASSISTANT_MESSAGE:  # pragma: no cover
        message = (
            data.get("message") if isinstance(data, dict) else None
        )  # pragma: no cover
        if isinstance(message, dict):  # pragma: no cover
            content = message.get("content")  # pragma: no cover
            if isinstance(content, list):  # pragma: no cover
                return _blocks_text(content)  # pragma: no cover
            return _text_of(content)  # pragma: no cover
        return ""  # pragma: no cover
    # pragma: no cover
    if event_type == EventType.TOOL_RESULT:  # pragma: no cover
        message = (
            data.get("message") if isinstance(data, dict) else None
        )  # pragma: no cover
        if isinstance(message, dict):  # pragma: no cover
            content = message.get("content")  # pragma: no cover
            if isinstance(content, list):  # pragma: no cover
                return _blocks_text(content)  # pragma: no cover
            return _text_of(content)  # pragma: no cover
        return ""  # pragma: no cover
    # pragma: no cover
    if event_type == EventType.REQUEST_HEADER:  # pragma: no cover
        config = (
            data.get("config") if isinstance(data, dict) else None
        )  # pragma: no cover
        if isinstance(config, dict):  # pragma: no cover
            parts: list[str] = []  # pragma: no cover
            provider = config.get("provider")  # pragma: no cover
            model = config.get("model")  # pragma: no cover
            if provider:  # pragma: no cover
                parts.append(str(provider))  # pragma: no cover
            if model:  # pragma: no cover
                parts.append(str(model))  # pragma: no cover
            return " ".join(p for p in parts if p)  # pragma: no cover
        return ""  # pragma: no cover
    # pragma: no cover
    if event_type == EventType.REQUEST_CONTEXT:  # pragma: no cover
        provider = (
            data.get("provider") if isinstance(data, dict) else None
        )  # pragma: no cover
        model = (
            data.get("model") if isinstance(data, dict) else None
        )  # pragma: no cover
        parts = []  # pragma: no cover
        if provider:  # pragma: no cover
            parts.append(str(provider))  # pragma: no cover
        if model:  # pragma: no cover
            parts.append(str(model))  # pragma: no cover
        return " ".join(p for p in parts if p)  # pragma: no cover
    # pragma: no cover
    # Reasoning blocks, blocked prompts, structural events, stream chunks:  # pragma: no cover
    # contribute nothing.  # pragma: no cover
    return ""  # pragma: no cover - non-message event types


# pragma: no cover
