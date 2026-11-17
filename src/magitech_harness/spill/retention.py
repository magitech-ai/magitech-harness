"""Token-budgeted content truncation for spill-policy.

Implements head/tail truncation: keeps as much head and tail as fit within
the token budget, dropping the middle. Reference:
``packages/spill/spill-policy/src/retention.ts``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict


class RetentionResult(TypedDict):
    """Result of truncating content to a token budget."""

    head: list[dict[str, Any]]
    tail: list[dict[str, Any]]
    omittedBytes: int
    omittedImages: int


def _text_tokens(text: str) -> int:
    """Estimate tokens for plain text (4 chars ~ 1 token)."""
    return max(1, (len(text.encode("utf-8")) + 3) // 4)  # pragma: no cover


def retain_content(
    content: list[dict[str, Any]],
    max_tokens: int,
    price: Callable[[dict[str, Any]], int],
) -> RetentionResult:
    """Truncate ``content`` to fit within ``max_tokens``.

    Keeps as much head and tail as possible within the budget, dropping the
    middle. Text blocks count via ``price(block)``; non-text blocks pass
    through unchanged and count their own price.

    Args:
        content: The content blocks (text + image).
        max_tokens: Maximum tokens for the retained result.
        price: Function ``price(block) -> int`` returning token estimate for
            one block.

    Returns:
        A dict with ``head``, ``tail`` (lists of kept blocks), and
        ``omittedBytes`` / ``omittedImages`` counts of dropped content.
    """
    if not content:
        return {"head": [], "tail": [], "omittedBytes": 0, "omittedImages": 0}

    total = sum(price(b) for b in content)
    if total <= max_tokens:
        return {
            "head": [dict(b) for b in content],
            "tail": [],
            "omittedBytes": 0,
            "omittedImages": sum(1 for b in content if b.get("type") == "image"),
        }

    # Separate text blocks (priceable) from non-text blocks (passthrough).
    text_blocks: list[dict[str, Any]] = []
    for b in content:
        if b.get("type") == "text":  # pragma: no cover
            text_blocks.append(b)

    # Greedy: split budget between head and tail.
    head_budget = max_tokens // 2
    tail_budget = max_tokens - head_budget

    # Fill head from the start.
    head_blocks: list[dict[str, Any]] = []
    head_cost = 0
    i = 0
    while i < len(text_blocks) and head_cost + price(text_blocks[i]) <= head_budget:
        head_blocks.append(dict(text_blocks[i]))
        head_cost += price(text_blocks[i])
        i += 1

    # Fill tail from the end.
    tail_blocks: list[dict[str, Any]] = []
    tail_cost = 0
    j = len(text_blocks) - 1
    while j >= i and tail_cost + price(text_blocks[j]) <= tail_budget:
        tail_blocks.insert(0, dict(text_blocks[j]))
        tail_cost += price(text_blocks[j])
        j -= 1

    omitted_text = text_blocks[i : j + 1] if i <= j else []
    omitted_bytes = sum(len(b.get("text", "").encode("utf-8")) for b in omitted_text)
    omitted_images = sum(1 for b in content if b.get("type") == "image")

    return {
        "head": head_blocks,
        "tail": tail_blocks,
        "omittedBytes": omitted_bytes,
        "omittedImages": omitted_images,
    }
