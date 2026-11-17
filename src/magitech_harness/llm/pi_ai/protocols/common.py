"""Shared HTTP, SSE, and chunk helpers for pi-ai protocol clients."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.attribution import attribution_headers
from magitech_harness.llm.types import FinishReason, StreamChunk, TokenUsage


def build_headers(
    *,
    api_key: str | None,
    extra: Mapping[str, str] | None = None,
    auth_header: str | None = None,
    auth_scheme: str = "Bearer",
) -> dict[str, str]:
    """Build provider request headers with harness attribution."""
    headers = dict(attribution_headers())
    headers["content-type"] = "application/json"
    if api_key and auth_header:
        headers[auth_header] = f"{auth_scheme} {api_key}" if auth_scheme else api_key
    if extra:
        headers.update(extra)
    return headers


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[dict[str, str]]:
    """Yield parsed SSE events as dicts with optional event/data fields."""
    event_name = "message"
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if data_lines:
                yield {"event": event_name, "data": "\n".join(data_lines)}
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].lstrip())
    if data_lines:
        yield {"event": event_name, "data": "\n".join(data_lines)}


def parse_json_data(data: str) -> Any | None:
    """Parse an SSE JSON payload, returning None for empty or terminal frames."""
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def usage_chunk(usage: TokenUsage) -> StreamChunk:
    """Build the usage chunk of a finished response."""
    return cast(StreamChunk, {"type": "usage", "usage": usage})


def finish_chunk(reason: FinishReason) -> StreamChunk:
    """Build the terminal finish chunk of a response."""
    return cast(StreamChunk, {"type": "finish", "reason": reason})


def token_usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: int | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> TokenUsage:
    """Build the token accounting of one model call."""
    usage: dict[str, Any] = {
        "inputTokens": int(input_tokens),
        "outputTokens": int(output_tokens),
    }
    total = (
        int(total_tokens)
        if total_tokens is not None
        else int(input_tokens) + int(output_tokens)
    )
    usage["totalTokens"] = total
    if cache_read_tokens:
        usage["cacheReadTokens"] = int(cache_read_tokens)
    if cache_write_tokens:
        usage["cacheWriteTokens"] = int(cache_write_tokens)
    if reasoning_tokens:
        usage["reasoningTokens"] = int(reasoning_tokens)
    return cast(TokenUsage, usage)


def map_finish_reason(kind: str | None) -> FinishReason:
    """Map a provider finish reason to the harness finish vocabulary."""
    if kind in (None, "stop", "end_turn", "STOP", "completed"):
        return {"kind": "stop"}
    if kind in ("length", "max_tokens", "MAX_TOKENS", "incomplete"):
        return {"kind": "max-tokens"}
    if kind in ("tool_calls", "tool_use", "function_call"):
        return {"kind": "tool-calls"}
    if kind in ("content_filter", "SAFETY", "RECITATION"):
        return {
            "kind": "error",
            "failure": {
                "message": f"provider stopped: {kind}",
                "code": "CONTENT_FILTER",
            },
        }
    return {"kind": "stop"}


def _extract_error_message(body: Any) -> str:
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            message = error.get("message") or error.get("type") or error.get("code")
            if isinstance(message, str) and message:
                return message
        message = body.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(body, str) and body:
        return body
    return "provider request failed"


def _failure_code(status: int, message: str) -> str:
    if status in (401, 403):
        return "AUTH"
    if status == 429:
        return "RATE_LIMIT"
    if status in (400, 413):
        return "INVALID_REQUEST"
    if 500 <= status <= 599:
        return "SERVER"
    if "timeout" in message.lower():
        return "TIMEOUT"
    return "PROVIDER_ERROR"


async def raise_for_status(response: httpx.Response) -> None:
    """Raise LlmError for a non-success provider response."""
    if response.is_success:
        return
    raw = await response.aread()
    text = raw.decode("utf-8", errors="replace")
    body: Any = text
    if text:
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = text
    message = _extract_error_message(body)
    request_id = response.headers.get("x-request-id") or response.headers.get(
        "request-id"
    )
    raise LlmError(
        f"pi-ai provider request failed ({response.status_code}): {message}",
        _failure_code(response.status_code, message),
        status=response.status_code,
        request_id=request_id,
    )


def flatten_text(content: Any) -> str:
    """Join the text blocks of a message content value."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
    return ""


def contains_image(content: Any) -> bool:
    """Report whether a message content value carries an image block."""
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, Mapping) and block.get("type") == "image" for block in content
    )


@dataclass
class _Block:
    """One assembled content block of a streamed response."""

    kind: str
    index: int
    text: str = ""
    tool_id: str = ""
    tool_name: str = ""
    arguments: str = ""


@dataclass
class StreamEmitter:
    """Turns provider stream events into the harness chunk vocabulary."""

    _next_index: int = 0
    _blocks: list[_Block] = field(default_factory=list)
    _tools: dict[str, _Block] = field(default_factory=dict)

    def _allocate(self, kind: str) -> _Block:
        block = _Block(kind=kind, index=self._next_index)
        self._next_index += 1
        self._blocks.append(block)
        return block

    def _text_block(self, kind: str) -> _Block | None:
        return next((b for b in self._blocks if b.kind == kind), None)

    def text(self, delta: str) -> list[StreamChunk]:
        """Emit a text delta, opening the text block on first use."""
        if not delta:
            return []
        block = self._text_block("text")
        chunks: list[StreamChunk] = []
        if block is None:
            block = self._allocate("text")
            chunks.append(
                cast(
                    StreamChunk,
                    {"type": "block-start", "index": block.index, "blockType": "text"},
                )
            )
        block.text += delta
        chunks.append(
            cast(
                StreamChunk, {"type": "text-delta", "index": block.index, "text": delta}
            )
        )
        return chunks

    def reasoning(self, delta: str) -> list[StreamChunk]:
        """Emit a reasoning delta, opening the reasoning block on first use."""
        if not delta:
            return []
        block = self._text_block("reasoning")
        chunks: list[StreamChunk] = []
        if block is None:
            block = self._allocate("reasoning")
            chunks.append(
                cast(
                    StreamChunk,
                    {
                        "type": "block-start",
                        "index": block.index,
                        "blockType": "reasoning",
                    },
                )
            )
        block.text += delta
        chunks.append(
            cast(
                StreamChunk,
                {"type": "reasoning-delta", "index": block.index, "text": delta},
            )
        )
        return chunks

    def tool_call(
        self,
        key: str,
        tool_id: str,
        name: str,
        arguments_delta: str,
        replace: bool = False,
    ) -> list[StreamChunk]:
        """Emit a tool-call delta for the tool call identified by *key*."""
        block = self._tools.get(key)
        chunks: list[StreamChunk] = []
        if block is None:
            block = self._allocate("tool-call")
            self._tools[key] = block
            chunks.append(
                cast(
                    StreamChunk,
                    {
                        "type": "block-start",
                        "index": block.index,
                        "blockType": "tool-call",
                    },
                )
            )
        if tool_id:
            block.tool_id = tool_id
        if name:
            block.tool_name = name
        if arguments_delta:
            if replace:
                block.arguments = arguments_delta
            else:
                block.arguments += arguments_delta
        chunk: dict[str, Any] = {
            "type": "tool-call-delta",
            "index": block.index,
            "id": block.tool_id,
            "argumentsDelta": arguments_delta,
        }
        if block.tool_name:
            chunk["name"] = block.tool_name
        chunks.append(cast(StreamChunk, chunk))
        return chunks

    def finalize(self) -> list[StreamChunk]:
        """Close every open block and emit them in stream order."""
        chunks: list[StreamChunk] = []
        for block in self._blocks:
            if block.kind == "text":
                body: dict[str, Any] = {"type": "text", "text": block.text}
            elif block.kind == "reasoning":
                body = {"type": "reasoning", "text": block.text}
            else:
                body = {
                    "type": "tool-call",
                    "id": block.tool_id,
                    "name": block.tool_name,
                    "arguments": block.arguments or "{}",
                }
            chunks.append(
                cast(
                    StreamChunk,
                    {"type": "block-end", "index": block.index, "block": body},
                )
            )
        return chunks
