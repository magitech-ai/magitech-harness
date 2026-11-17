"""OpenAI Responses streaming protocol client."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import httpx

from magitech_harness.agent.cancellation import AbortError
from magitech_harness.llm.pi_ai.catalog import ModelSpec, resolve_reasoning_effort
from magitech_harness.llm.pi_ai.protocols.common import (
    StreamEmitter,
    build_headers,
    contains_image,
    finish_chunk,
    flatten_text,
    iter_sse_events,
    map_finish_reason,
    parse_json_data,
    raise_for_status,
    token_usage,
    usage_chunk,
)


def _assistant_items(content: Any) -> list[dict[str, Any]]:
    """Convert assistant history blocks into Responses output items."""
    items: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, Mapping):
            continue
        block_type = block.get("type")
        if block_type == "text":
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": str(block.get("text", ""))}
                    ],
                }
            )
        elif block_type == "tool-call":
            arguments = block.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            items.append(
                {
                    "type": "function_call",
                    "call_id": str(block.get("id", "")),
                    "name": str(block.get("name", "")),
                    "arguments": arguments,
                }
            )
    return items


def _convert_input(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert harness history into Responses input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        if role == "system":
            continue
        if role == "developer":
            text = flatten_text(content)
            if text:
                items.append(
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": text}],
                    }
                )
        elif role == "user":
            if contains_image(content):
                raise RuntimeError("image input unsupported")
            items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": flatten_text(content)}],
                }
            )
        elif role == "assistant":
            items.extend(_assistant_items(content))
        elif role == "tool":
            tool_call_id = str(
                message.get("toolCallId") or message.get("tool_call_id") or ""
            )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": flatten_text(content) or "(no output)",
                }
            )
    return items


def _convert_tools(tools: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert harness tool schemas into Responses tool definitions."""
    if not tools:
        return []
    return [
        {
            "type": "function",
            "name": str(tool.get("name", "")),
            "description": str(tool.get("description", "")),
            "parameters": tool.get("parameters")
            or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]


def _system_text(
    messages: Sequence[Mapping[str, Any]], system: str | None
) -> str | None:
    """Join the request-level and history system prompts."""
    parts: list[str] = []
    if system:
        parts.append(system)
    parts.extend(
        text
        for text in (
            flatten_text(message.get("content", []))
            for message in messages
            if message.get("role") == "system"
        )
        if text
    )
    joined = "\n\n".join(parts)
    return joined or None


def _build_body(spec: ModelSpec, options: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Responses request body."""
    messages = cast(Sequence[Mapping[str, Any]], options.get("messages") or [])
    body: dict[str, Any] = {
        "model": spec.id,
        "input": _convert_input(messages),
        "stream": True,
    }
    instructions = _system_text(messages, cast(str | None, options.get("system")))
    if instructions:
        body["instructions"] = instructions
    if options.get("temperature") is not None:
        body["temperature"] = options["temperature"]
    if options.get("maxTokens") is not None:
        body["max_output_tokens"] = options["maxTokens"]
    tools = _convert_tools(
        cast(Sequence[Mapping[str, Any]] | None, options.get("tools"))
    )
    if tools:
        body["tools"] = tools
    reasoning = resolve_reasoning_effort(
        spec, cast(str | None, options.get("reasoningEffort"))
    )
    if reasoning and reasoning != "off":
        mapped = (spec.thinking_level_map or {}).get(reasoning)
        body["reasoning"] = {"effort": mapped if isinstance(mapped, str) else reasoning}
    return body


def _event_type(payload: Mapping[str, Any], event: Mapping[str, str]) -> str:
    """Resolve the type of one Responses stream event."""
    return str(payload.get("type") or event.get("event") or "")


def _chunks_of(
    payload: Mapping[str, Any],
    event: Mapping[str, str],
    emitter: StreamEmitter,
) -> tuple[list[dict[str, Any]], str | None]:
    """Translate one Responses event into harness chunks."""
    chunks: list[dict[str, Any]] = []
    event_type = _event_type(payload, event)
    if event_type == "response.output_text.delta":
        chunks.extend(emitter.text(str(payload.get("delta", ""))))
    elif event_type in (
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
    ):
        chunks.extend(emitter.reasoning(str(payload.get("delta", ""))))
    elif event_type == "response.output_item.added":
        item = cast(Mapping[str, Any], payload.get("item") or {})
        if item.get("type") == "function_call":
            key = str(payload.get("output_index") or 0)
            chunks.extend(
                emitter.tool_call(
                    key,
                    str(item.get("call_id") or item.get("id") or ""),
                    str(item.get("name") or ""),
                    "",
                )
            )
    elif event_type == "response.function_call_arguments.delta":
        key = str(payload.get("output_index") or 0)
        chunks.extend(emitter.tool_call(key, "", "", str(payload.get("delta", ""))))
    elif event_type == "response.function_call_arguments.done":
        key = str(payload.get("output_index") or 0)
        chunks.extend(
            emitter.tool_call(
                key, "", "", str(payload.get("arguments") or ""), replace=True
            )
        )
    elif event_type in ("response.completed", "response.incomplete"):
        response = cast(Mapping[str, Any], payload.get("response") or {})
        status = response.get("status") or (
            "completed" if event_type.endswith("completed") else "incomplete"
        )
        usage = response.get("usage")
        if isinstance(usage, Mapping):
            input_details = cast(
                Mapping[str, Any], usage.get("input_tokens_details") or {}
            )
            output_details = cast(
                Mapping[str, Any], usage.get("output_tokens_details") or {}
            )
            chunks.append(
                usage_chunk(
                    token_usage(
                        input_tokens=int(usage.get("input_tokens") or 0),
                        output_tokens=int(usage.get("output_tokens") or 0),
                        total_tokens=int(usage.get("total_tokens") or 0),
                        cache_read_tokens=int(input_details.get("cached_tokens") or 0),
                        reasoning_tokens=int(
                            output_details.get("reasoning_tokens") or 0
                        ),
                    )
                )
            )
        return chunks, str(status)
    elif event_type == "response.failed":
        response = cast(Mapping[str, Any], payload.get("response") or {})
        error = response.get("error")
        message = "response failed"
        if isinstance(error, Mapping):
            message = str(error.get("message") or error.get("code") or message)
        raise RuntimeError(message)
    return chunks, None


async def stream_openai_responses(
    spec: ModelSpec,
    options: Mapping[str, Any],
    *,
    api_key: str | None,
    headers: Mapping[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream harness chunks from an OpenAI Responses endpoint."""
    url = f"{spec.base_url}/responses"
    request_headers = build_headers(
        api_key=api_key,
        extra=headers,
        auth_header="authorization",
        auth_scheme="Bearer",
    )
    body = _build_body(spec, options)
    signal = options.get("signal")
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=None)
    try:
        async with http.stream(
            "POST", url, headers=request_headers, json=body
        ) as response:
            await raise_for_status(response)
            emitter = StreamEmitter()
            finish_reason: str | None = None
            async for event in iter_sse_events(response):
                if signal is not None and getattr(signal, "is_aborted", False):
                    raise AbortError("Operation was aborted.")
                payload = parse_json_data(event.get("data", ""))
                if not isinstance(payload, Mapping):
                    continue
                chunks, reason = _chunks_of(payload, event, emitter)
                for chunk in chunks:
                    yield chunk
                if reason is not None:
                    finish_reason = reason
            for chunk in emitter.finalize():
                yield chunk
            yield finish_chunk(map_finish_reason(finish_reason))
    finally:
        if owns_client:
            await http.aclose()
