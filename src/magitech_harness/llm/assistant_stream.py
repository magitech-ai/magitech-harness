"""Compact timed stream representation and record-level stream readers."""

from __future__ import annotations

import copy
import json
import time as _time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

from magitech_harness.llm.assembler import BlockAssembler
from magitech_harness.llm.types import ModelChunk, StreamChunk
from magitech_harness.llm.values import (
    FrozenDict,
    assert_never,
    deep_freeze,
    is_non_negative_safe_integer,
    is_safe_integer,
    snapshot_json_value,
)

RawStreamChunkType = Literal["block-start", "block-end", "usage", "finish"]


class TimedStreamChunk(TypedDict):
    """One model chunk paired with its timestamp."""

    time: int
    chunk: StreamChunk


AssistantStreamRecord = dict[str, Any]
AssistantStreamRun = dict[str, Any]


def _safe_time(value: Any) -> int:
    if not is_safe_integer(value):
        raise TypeError(f"Assistant stream time must be a safe integer, got {value!r}")
    return value


def _safe_index(value: Any, label: str) -> int:
    if not is_non_negative_safe_integer(value):
        raise TypeError(f"{label} index must be a non-negative safe integer")
    return value


def _snapshot_chunk(chunk: Any) -> StreamChunk:
    snapped = snapshot_json_value(chunk)
    if snapped is None or not isinstance(snapped, dict):
        raise TypeError("Assistant stream chunk must be losslessly JSON-serializable")
    return snapped


def _safe_gap(previous: int, nxt: int) -> int | None:
    gap = nxt - previous
    if is_safe_integer(gap) and previous + gap == nxt:
        return gap
    return None


class AssistantStreamAccumulator:
    """Compacts one attempt's timed chunks (parity or legacy ModelChunk)."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._legacy_only = True

    def push(self, value: ModelChunk | Mapping[str, Any]) -> TimedStreamChunk | None:
        """Append one ModelChunk (legacy) or TimedStreamChunk (parity)."""
        if isinstance(value, ModelChunk):
            self._records.append(
                {
                    "type": "chunk",
                    "time": int(_time.time() * 1000),
                    "chunk": value.model_dump(),
                }
            )
            return None

        self._legacy_only = False
        safe_t = _safe_time(value["time"])
        chunk = _snapshot_chunk(value["chunk"])
        timed = deep_freeze({"time": safe_t, "chunk": copy.deepcopy(chunk)})
        previous = self._records[-1] if self._records else None
        chunk_type = chunk.get("type")

        if chunk_type in ("text-delta", "reasoning-delta"):
            idx = _safe_index(chunk.get("index"), chunk_type)
            text = chunk.get("text")
            if not isinstance(text, str):
                raise TypeError(f"{chunk_type} text must be a string")
            rec_type = (
                "text-chunks" if chunk_type == "text-delta" else "reasoning-chunks"
            )
            gap = (
                _safe_gap(previous["lastTime"], safe_t)
                if previous is not None and previous["type"] == rec_type
                else None
            )
            if (
                previous is not None
                and previous["type"] == rec_type
                and previous["index"] == idx
                and gap is not None
            ):
                previous["dt"].append(gap)
                previous["texts"].append(text)
                previous["lastTime"] = safe_t
            else:
                self._records.append(
                    {
                        "type": rec_type,
                        "time0": safe_t,
                        "index": idx,
                        "dt": [],
                        "texts": [text],
                        "lastTime": safe_t,
                    }
                )
            return timed  # type: ignore[return-value]

        if chunk_type == "tool-call-delta":
            idx = _safe_index(chunk.get("index"), chunk_type)
            call_id = chunk.get("id")
            if not isinstance(call_id, str):
                raise TypeError("tool-call-delta id must be a string")
            if "name" in chunk and not isinstance(chunk["name"], str):
                raise TypeError("tool-call-delta name must be a string")
            args_delta = chunk.get("argumentsDelta")
            if not isinstance(args_delta, str):
                raise TypeError("tool-call-delta argumentsDelta must be a string")
            if len(call_id) == 0 or chunk.get("name") == "":
                self._records.append({"type": "chunk", "time": safe_t, "chunk": chunk})
                return timed  # type: ignore[return-value]
            gap = (
                _safe_gap(previous["lastTime"], safe_t)
                if previous is not None and previous["type"] == "tool-call-chunks"
                else None
            )
            same_name = (
                previous is not None
                and previous["type"] == "tool-call-chunks"
                and ("name" in previous) == ("name" in chunk)
                and previous.get("name") == chunk.get("name")
            )
            if (
                previous is not None
                and previous["type"] == "tool-call-chunks"
                and previous["index"] == idx
                and previous["id"] == call_id
                and same_name
                and gap is not None
            ):
                previous["dt"].append(gap)
                previous["args"].append(args_delta)
                previous["lastTime"] = safe_t
            else:
                record: dict[str, Any] = {
                    "type": "tool-call-chunks",
                    "time0": safe_t,
                    "index": idx,
                    "dt": [],
                    "id": call_id,
                    "args": [args_delta],
                    "lastTime": safe_t,
                }
                if "name" in chunk:
                    record["name"] = chunk["name"]
                self._records.append(record)
            return timed  # type: ignore[return-value]

        if chunk_type in ("block-start", "block-end", "usage", "finish"):
            self._records.append({"type": "chunk", "time": safe_t, "chunk": chunk})
            return timed  # type: ignore[return-value]

        assert_never(chunk, "AssistantStreamAccumulator.push")

    def snapshot(self) -> list[dict[str, Any]]:
        """Return detached durable records (frozen for parity streams)."""
        if self._legacy_only and self._records:
            return copy.deepcopy(self._records)
        records: list[dict[str, Any]] = []
        for record in self._records:
            if record["type"] == "chunk":
                records.append(copy.deepcopy(record))
            else:
                durable = {k: v for k, v in record.items() if k != "lastTime"}
                if durable["type"] == "tool-call-chunks":
                    durable["dt"] = list(durable["dt"])
                    durable["args"] = list(durable["args"])
                else:
                    durable["dt"] = list(durable["dt"])
                    durable["texts"] = list(durable["texts"])
                records.append(durable)
        return deep_freeze(records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return len(self._records) > 0


def _exact_keys(record: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    if len(record) != len(keys) or not all(k in record for k in keys):
        joined = ", ".join(keys)
        raise TypeError(
            f"{label} Assistant stream record must contain exactly {joined}"
        )


def _string_array(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{label} must be a string array")
    return list(value)


def _validate_run(record: Mapping[str, Any], members: int, label: str) -> None:
    _safe_time(record["time0"])
    _safe_index(record["index"], label)
    dt = record["dt"]
    if not isinstance(dt, list) or not all(is_safe_integer(val) for val in dt):
        raise TypeError(f"{label} dt must contain safe integers")
    if len(dt) != members - 1:
        raise TypeError(f"{label} dt length must be one less than its members")
    cur_time = int(record["time0"])
    for gap in dt:
        cur_time += int(gap)
        if not is_safe_integer(cur_time):
            raise TypeError(f"{label} member times must stay safe integers")


def _validate_record(value: Any) -> dict[str, Any]:
    if type(value) not in (dict, FrozenDict):
        raise TypeError("Assistant stream record must be an object")
    rec_type = value.get("type")
    if rec_type in ("text-chunks", "reasoning-chunks"):
        _exact_keys(value, ("type", "time0", "index", "dt", "texts"), rec_type)
        texts = _string_array(value["texts"], f"{rec_type} texts")
        if len(texts) == 0:
            raise TypeError(f"{rec_type} texts must be non-empty")
        _validate_run(value, len(texts), rec_type)
        return dict(value)
    if rec_type == "tool-call-chunks":
        keys = (
            ("type", "time0", "index", "dt", "id", "name", "args")
            if "name" in value
            else ("type", "time0", "index", "dt", "id", "args")
        )
        _exact_keys(value, keys, rec_type)
        args = _string_array(value["args"], "tool-call-chunks args")
        if len(args) == 0:
            raise TypeError("tool-call-chunks args must be non-empty")
        call_id = value["id"]
        if not isinstance(call_id, str) or len(call_id) == 0:
            raise TypeError("tool-call-chunks id must be a non-empty string")
        if "name" in value:
            name = value["name"]
            if not isinstance(name, str) or len(name) == 0:
                raise TypeError("tool-call-chunks name must be a non-empty string")
        _validate_run(value, len(args), rec_type)
        return dict(value)
    if rec_type == "chunk":
        _exact_keys(value, ("type", "time", "chunk"), "chunk")
        safe_t = _safe_time(value["time"])
        raw_chunk = value["chunk"]
        if type(raw_chunk) not in (dict, FrozenDict):
            raise TypeError("Assistant stream raw chunk must be a lossless JSON object")
        try:
            snapped = _snapshot_chunk(raw_chunk)
        except Exception as err:
            raise TypeError(
                "Assistant stream raw chunk must be a lossless JSON object"
            ) from err
        return deep_freeze({"type": "chunk", "time": safe_t, "chunk": snapped})
    raise TypeError(f"Unsupported Assistant stream record {json.dumps(rec_type)}")


def expand_assistant_stream(
    stream: Sequence[AssistantStreamRecord],
) -> list[TimedStreamChunk]:
    """Expand and validate compact records into timed chunk sequence."""
    chunks: list[TimedStreamChunk] = []
    for candidate in stream:
        record = _validate_record(candidate)
        if record["type"] == "chunk":
            chunks.append({"time": record["time"], "chunk": record["chunk"]})
            continue
        members = (
            record["args"] if record["type"] == "tool-call-chunks" else record["texts"]
        )
        cur_time = record["time0"]
        for idx, member in enumerate(members):
            if idx > 0:
                cur_time += record["dt"][idx - 1]
            if record["type"] == "text-chunks":
                chunk: StreamChunk = {
                    "type": "text-delta",
                    "index": record["index"],
                    "text": member,
                }
            elif record["type"] == "reasoning-chunks":
                chunk = {
                    "type": "reasoning-delta",
                    "index": record["index"],
                    "text": member,
                }
            else:
                chunk = {
                    "type": "tool-call-delta",
                    "index": record["index"],
                    "id": record["id"],
                    "argumentsDelta": member,
                }
                if "name" in record:
                    chunk["name"] = record["name"]
            chunks.append({"time": cur_time, "chunk": chunk})
    return chunks


def _has_non_whitespace(text: str) -> bool:
    return any(not ch.isspace() for ch in text)


def _block_is_visible(block: Mapping[str, Any]) -> bool:
    btype = block.get("type")
    if btype == "tool-call":
        return False
    if btype in ("text", "reasoning"):
        return _has_non_whitespace(str(block.get("text", "")))
    return True


def is_token_delta(chunk: Mapping[str, Any]) -> bool:
    """Return True if chunk carries non-empty text/reasoning/tool-call token output."""
    ctype = chunk.get("type")
    if ctype in ("text-delta", "reasoning-delta"):
        return chunk.get("text") != ""
    if ctype == "tool-call-delta":
        return chunk.get("argumentsDelta") != "" or chunk.get("name") is not None
    return False


def is_visible_chunk(chunk: Mapping[str, Any]) -> bool:
    """Return True if chunk contributes reader-visible transcript content."""
    ctype = chunk.get("type")
    if ctype in ("text-delta", "reasoning-delta"):
        return _has_non_whitespace(str(chunk.get("text", "")))
    if ctype == "block-start":
        return chunk.get("blockType") not in ("text", "reasoning", "tool-call")
    if ctype == "block-end":
        return _block_is_visible(chunk["block"])
    return False


def chunk_has_visible_text(chunk: Mapping[str, Any]) -> bool:
    """Return True if chunk contributes non-whitespace visible text."""
    ctype = chunk.get("type")
    if ctype == "text-delta":
        return _has_non_whitespace(str(chunk.get("text", "")))
    if ctype == "block-end":
        block = chunk["block"]
        return block.get("type") == "text" and _has_non_whitespace(
            str(block.get("text", ""))
        )
    return False


def _first_run_member_time(run: Mapping[str, Any], predicate: Any) -> int | None:
    fragments = run["args"] if run["type"] == "tool-call-chunks" else run["texts"]
    cur_time = int(run["time0"])
    for idx, fragment in enumerate(fragments):
        if idx > 0:
            cur_time += int(run["dt"][idx - 1])
        if predicate(fragment):
            return cur_time
    return None


def run_first_token_time(run: Mapping[str, Any]) -> int | None:
    """Return reconstructed timestamp of the first token-bearing member in a run."""
    if run["type"] == "tool-call-chunks" and run.get("name") is not None:
        return int(run["time0"])
    return _first_run_member_time(run, lambda frag: frag != "")


def run_first_visible_time(run: Mapping[str, Any]) -> int | None:
    """Return reconstructed timestamp of the first non-whitespace visible member."""
    if run["type"] == "tool-call-chunks":
        return None
    return _first_run_member_time(run, _has_non_whitespace)


def assistant_stream_first_token_time(
    stream: Sequence[AssistantStreamRecord],
) -> int | None:
    """Return timestamp of the first token-bearing chunk in a compact stream."""
    for record in stream:
        if record["type"] == "chunk":
            time_val = record["time"] if is_token_delta(record["chunk"]) else None
        else:
            time_val = run_first_token_time(record)
        if time_val is not None:
            return int(time_val)
    return None


def assistant_stream_has_visible_content(
    stream: Sequence[AssistantStreamRecord],
) -> bool:
    """Return True if a compact stream carries any reader-visible content."""
    return any(
        is_visible_chunk(rec["chunk"])
        if rec["type"] == "chunk"
        else run_first_visible_time(rec) is not None
        for rec in stream
    )


def assistant_stream_has_visible_text(
    stream: Sequence[AssistantStreamRecord],
) -> bool:
    """Return True if a compact stream carries non-whitespace visible text."""
    return any(
        any(_has_non_whitespace(t) for t in rec["texts"])
        if rec["type"] == "text-chunks"
        else rec["type"] == "chunk" and chunk_has_visible_text(rec["chunk"])
        for rec in stream
    )


def last_assistant_stream_chunk(
    stream: Sequence[AssistantStreamRecord], chunk_type: RawStreamChunkType
) -> StreamChunk | None:
    """Return the last raw chunk of *chunk_type* in a compact stream, or None."""
    for record in reversed(stream):
        if record["type"] == "chunk" and record["chunk"].get("type") == chunk_type:
            return record["chunk"]  # type: ignore[no-any-return]
    return None


def assistant_stream_chunks(
    stream: Sequence[AssistantStreamRecord], chunk_type: RawStreamChunkType
) -> list[StreamChunk]:
    """Return every raw chunk of *chunk_type* in stream order."""
    return [
        record["chunk"]
        for record in stream
        if record["type"] == "chunk" and record["chunk"].get("type") == chunk_type
    ]


def join_assistant_stream_text(stream: Sequence[AssistantStreamRecord]) -> str:
    """Join all text-delta fragments from compact records in stream order."""
    parts: list[str] = []
    for record in stream:
        if record["type"] == "text-chunks":
            parts.append("".join(record["texts"]))
        elif record["type"] == "chunk" and record["chunk"].get("type") == "text-delta":
            parts.append(record["chunk"]["text"])
    return "".join(parts)


def assemble_assistant_stream(
    stream: Sequence[AssistantStreamRecord],
    assembler: BlockAssembler | None = None,
) -> BlockAssembler:
    """Feed a compact stream into BlockAssembler without expanding runs."""
    target = assembler if assembler is not None else BlockAssembler()
    for record in stream:
        rec_type = record.get("type")
        if rec_type == "chunk":
            target.push(record["chunk"])
        elif rec_type == "text-chunks":
            target.push(
                {
                    "type": "text-delta",
                    "index": record["index"],
                    "text": "".join(record["texts"]),
                }
            )
        elif rec_type == "reasoning-chunks":
            target.push(
                {
                    "type": "reasoning-delta",
                    "index": record["index"],
                    "text": "".join(record["texts"]),
                }
            )
        elif rec_type == "tool-call-chunks":
            delta: StreamChunk = {
                "type": "tool-call-delta",
                "index": record["index"],
                "id": record["id"],
                "argumentsDelta": "".join(record["args"]),
            }
            if "name" in record:
                delta["name"] = record["name"]
            target.push(delta)
        else:
            assert_never(record, "assemble_assistant_stream")
    return target


expandAssistantStream = expand_assistant_stream
isTokenDelta = is_token_delta
isVisibleChunk = is_visible_chunk
chunkHasVisibleText = chunk_has_visible_text
runFirstTokenTime = run_first_token_time
runFirstVisibleTime = run_first_visible_time
assistantStreamFirstTokenTime = assistant_stream_first_token_time
assistantStreamHasVisibleContent = assistant_stream_has_visible_content
assistantStreamHasVisibleText = assistant_stream_has_visible_text
lastAssistantStreamChunk = last_assistant_stream_chunk
assistantStreamChunks = assistant_stream_chunks
joinAssistantStreamText = join_assistant_stream_text
assembleAssistantStream = assemble_assistant_stream
