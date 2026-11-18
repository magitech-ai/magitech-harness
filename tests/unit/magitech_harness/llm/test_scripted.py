"""Unit tests for the config-driven scripted LLM adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from magitech_harness.llm.adapter_failure import LlmError
from magitech_harness.llm.assembler import BlockAssembler
from magitech_harness.llm.scripted import (
    ScriptedLlmAdapter,
    ScriptSpec,
    load_script,
)

TOOL_TURN = (
    {
        "type": "text-delta",
        "index": 0,
        "text": "Computing. ",
    },
    {
        "type": "tool-call-delta",
        "index": 1,
        "id": "call_mul_1",
        "name": "multiply",
        "argumentsDelta": '{"a": 6, "b": 7}',
    },
    {"type": "finish", "reason": {"kind": "tool-calls"}},
)
TEXT_TURN = (
    {"type": "text-delta", "index": 0, "text": "Product is 42."},
    {"type": "finish", "reason": {"kind": "stop"}},
)


@pytest.mark.asyncio
async def test_scripted_adapter_streams_turns_in_order() -> None:
    adapter = ScriptedLlmAdapter(
        ScriptSpec("scripted", "script", (TOOL_TURN, TEXT_TURN))
    )
    first = [chunk async for chunk in adapter.stream({})]
    second = [chunk async for chunk in adapter.stream({})]
    assert first == [dict(c) for c in TOOL_TURN]
    assert second == [dict(c) for c in TEXT_TURN]
    models = await adapter.list_models("scripted")
    assert models == [{"provider": "scripted", "id": "script", "name": "script"}]


@pytest.mark.asyncio
async def test_scripted_adapter_fails_loud_when_exhausted() -> None:
    adapter = ScriptedLlmAdapter(ScriptSpec("scripted", "script", (TEXT_TURN,)))
    [chunk async for chunk in adapter.stream({})]
    with pytest.raises(LlmError, match="exhausted"):
        [chunk async for chunk in adapter.stream({})]


def test_scripted_chunks_assemble_into_tool_call_block() -> None:
    asm = BlockAssembler()
    for chunk in TOOL_TURN:
        asm.push(chunk)
    blocks = asm.blocks()
    assert {
        "type": "tool-call",
        "id": "call_mul_1",
        "name": "multiply",
        "arguments": '{"a": 6, "b": 7}',
    } in blocks
    assert asm.finish == {"kind": "tool-calls"}


def test_load_script_parses_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "turns.json"
    path.write_text(
        json.dumps(
            {
                "provider": "scripted",
                "model": "script",
                "turns": [list(TOOL_TURN), list(TEXT_TURN)],
            }
        ),
        encoding="utf-8",
    )
    spec = load_script(path)
    assert spec.provider == "scripted"
    assert spec.model == "script"
    assert spec.turns == (TOOL_TURN, TEXT_TURN)


def test_load_script_defaults_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "defaults.json"
    path.write_text(json.dumps({"turns": [list(TEXT_TURN)]}), encoding="utf-8")
    spec = load_script(path)
    assert (spec.provider, spec.model) == ("scripted", "script")

    for bad, match in [
        ("not json", "not valid JSON"),
        ('["list"]', "JSON object"),
        (json.dumps({"turns": []}), "non-empty list"),
        (json.dumps({"turns": ["no"]}), "list of chunks"),
        (json.dumps({"turns": [[42]]}), "string type"),
        (json.dumps({"turns": [list(TEXT_TURN)], "provider": ""}), "non-empty string"),
        (json.dumps({"turns": [list(TEXT_TURN)], "model": 1}), "non-empty string"),
    ]:
        bad_path = tmp_path / "bad.json"
        bad_path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_script(bad_path)
