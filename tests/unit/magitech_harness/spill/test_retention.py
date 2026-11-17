"""Unit tests for spill-policy retention (head/tail truncation)."""

from __future__ import annotations

from typing import Any

from magitech_harness.spill.retention import retain_content


def _price(block: dict[str, Any]) -> int:
    """Price function: text blocks = char count // 4, others = 0."""
    if block.get("type") == "text":
        text = block.get("text", "")
        return max(1, len(text) // 4)
    return 0


def test_retain_content_under_budget() -> None:
    """Content within budget passes through unchanged."""
    content = [{"type": "text", "text": "hello world"}]
    result = retain_content(content, 100, _price)
    assert result["head"] == [{"type": "text", "text": "hello world"}]
    assert result["tail"] == []
    assert result["omittedBytes"] == 0
    assert result["omittedImages"] == 0


def test_retain_content_truncates_to_head() -> None:
    """Large text is truncated to head within budget."""
    text_blocks = [{"type": "text", "text": f"line {i}"} for i in range(10)]
    result = retain_content(text_blocks, 9, _price)
    assert len(result["head"]) > 0
    assert result["omittedBytes"] > 0


def test_retain_content_head_and_tail() -> None:
    """Both head and tail blocks are kept when they fit."""
    blocks = [{"type": "text", "text": f"line {i}"} for i in range(20)]
    result = retain_content(blocks, 15, _price)
    # With budget 15, head takes some, tail takes some
    assert len(result["head"]) > 0
    assert len(result["tail"]) > 0


def test_retain_content_preserves_non_text() -> None:
    """Non-text blocks pass through unchanged."""
    content = [
        {"type": "text", "text": "some text here"},
        {"type": "image", "attachment": {"attachmentId": "img-1"}},
        {"type": "text", "text": "more text"},
    ]
    result = retain_content(content, 1000, _price)
    assert len(result["head"]) == 3
    assert result["head"][0]["type"] == "text"
    assert result["head"][1]["type"] == "image"
    assert result["head"][2]["type"] == "text"


def test_retain_content_empty() -> None:
    """Empty content returns empty result."""
    result = retain_content([], 100, _price)
    assert result["head"] == []
    assert result["tail"] == []
    assert result["omittedBytes"] == 0
    assert result["omittedImages"] == 0


def test_retain_content_single_block_fits() -> None:
    """Single small block fits in budget."""
    content = [{"type": "text", "text": "hi"}]
    result = retain_content(content, 100, _price)
    assert result["head"] == [{"type": "text", "text": "hi"}]
    assert result["tail"] == []
    assert result["omittedBytes"] == 0


def test_retain_content_all_omitted() -> None:
    """When even head doesn't fit, all text is omitted."""
    content = [{"type": "text", "text": "a very long line that exceeds budget"}]
    result = retain_content(content, 1, _price)
    assert result["head"] == []
    assert result["tail"] == []
    assert result["omittedBytes"] > 0
