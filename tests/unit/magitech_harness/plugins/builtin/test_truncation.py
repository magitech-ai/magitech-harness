"""Unit tests for truncation middleware and functions."""

from __future__ import annotations

import pytest
from anima import Context

from magitech_harness.plugins.builtin.truncation import (
    _tail_suffix,
    apply,
    format_size,
    truncate_head,
    truncate_tail,
)
from magitech_harness.session.events import ToolCallData
from magitech_harness.tools.definition import define_tool
from magitech_harness.tools.service import ToolService


def test_format_size() -> None:
    """Verifies byte size formatting across units."""
    assert format_size(500) == "500B"
    assert format_size(2048) == "2.0KB"
    assert format_size(2 * 1024 * 1024) == "2.0MB"


def test_tail_suffix() -> None:
    """Verifies _tail_suffix edge cases."""
    # Under budget
    assert _tail_suffix("hello", 100) == "hello"
    # Zero budget
    assert _tail_suffix("hello", 0) == ""


def test_truncate_head_empty_and_short() -> None:
    """Verifies truncate_head returns untouched result when under limits."""
    empty_res = truncate_head("")
    assert not empty_res.truncated
    assert empty_res.text == ""

    short_text = "line1\nline2\n"
    res = truncate_head(short_text, max_lines=10, max_bytes=100)
    assert not res.truncated
    assert res.text == short_text
    assert res.total_lines == 2


def test_truncate_head_lines_exceeded() -> None:
    """Verifies truncate_head truncates when max_lines is exceeded."""
    text = "line1\nline2\nline3\nline4\n"
    res = truncate_head(text, max_lines=2, max_bytes=1000)
    assert res.truncated
    assert res.strategy == "head"
    assert res.shown_start == 1
    assert res.shown_end == 2
    assert res.text == "line1\nline2"


def test_truncate_head_bytes_exceeded() -> None:
    """Verifies truncate_head truncates when max_bytes is exceeded."""
    # Break during lines loop when some lines already kept
    text = "line1\nline2\nline3\n"
    res = truncate_head(text, max_lines=10, max_bytes=8)
    assert res.truncated
    assert res.shown_end == 1
    assert res.text == "line1"

    # Break when 0 lines kept
    res_zero = truncate_head(text, max_lines=10, max_bytes=2)
    assert res_zero.truncated
    assert res_zero.shown_end is None
    assert res_zero.text == ""


def test_truncate_tail_short_and_exceeded() -> None:
    """Verifies truncate_tail preserves end of content."""
    text = "a\nb\nc\nd\n"
    short_res = truncate_tail(text, max_lines=10, max_bytes=100)
    assert not short_res.truncated
    assert short_res.text == text

    line_res = truncate_tail(text, max_lines=2, max_bytes=100)
    assert line_res.truncated
    assert line_res.strategy == "tail"
    assert line_res.shown_start == 3
    assert line_res.shown_end == 4
    assert line_res.text == "c\nd"

    # Break when some lines already kept
    byte_res = truncate_tail(text, max_lines=10, max_bytes=2)
    assert byte_res.truncated
    assert byte_res.text == "d"

    # Single large line suffix fallback
    huge_line = "abcdefghijklmnopqrstuvwxyz"
    tail_res = truncate_tail(huge_line, max_lines=10, max_bytes=5)
    assert tail_res.truncated
    assert tail_res.text == "vwxyz"

    # Bytes exceeded with 0 bytes budget
    zero_res = truncate_tail(huge_line, max_lines=10, max_bytes=0)
    assert zero_res.truncated
    assert zero_res.text == ""


@pytest.mark.asyncio
async def test_truncation_plugin_waterfall_integration() -> None:
    """Verifies truncation plugin attaches to tools/post-execute waterfall."""
    ctx = Context()
    ctx.plugin(ToolService)
    ctx.plugin(apply, max_lines=2, max_bytes=50)
    await ctx.flush()

    # Tool returning 4 lines (exceeds max_lines=2)
    tool = define_tool(
        name="verbose_tool",
        description="Returns lots of text",
        execute=lambda args: "line1\nline2\nline3\nline4",
    )
    ctx.tools.register(tool)

    call_data = ToolCallData(call_id="call-1", name="verbose_tool", arguments={})
    res = await ctx.tools.execute(call_data)

    assert not res.is_error
    assert "[Output truncated: showing 2 of 4 lines" in res.content
    assert "line1\nline2" in res.content

    # Tool returning short text (not truncated)
    short_tool = define_tool(
        name="short_tool",
        description="Returns short text",
        execute=lambda args: "short",
    )
    ctx.tools.register(short_tool)
    call_short = ToolCallData(call_id="call-short", name="short_tool", arguments={})
    res_short = await ctx.tools.execute(call_short)
    assert res_short.content == "short"

    # Tool returning non-string is not modified
    obj_tool = define_tool(
        name="obj_tool",
        description="Returns non-string",
        execute=lambda args: {"count": 42},
    )
    ctx.tools.register(obj_tool)
    call_obj = ToolCallData(call_id="call-2", name="obj_tool", arguments={})
    res_obj = await ctx.tools.execute(call_obj)
    assert res_obj.content == {"count": 42}
