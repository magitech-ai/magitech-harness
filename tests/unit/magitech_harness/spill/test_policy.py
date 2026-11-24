"""Integration tests for spill-policy component."""

from __future__ import annotations

import pytest
from animas import Context

from magitech_harness.spill.local import LocalSpillStore
from magitech_harness.spill.policy import apply


class MockLLM:
    """Mock LLM service with image_request_pricing."""

    def __init__(self, pricing_result=None):
        self.image_request_pricing = pricing_result
        self.providers = {"test": {}}
        self.active_provider = "test"


@pytest.mark.asyncio
async def test_policy_full_flow_with_llm_and_spill() -> None:
    """Full flow: large text truncated and spilled."""
    ctx = Context()
    ctx.use(LocalSpillStore, {"root": "/tmp/spill-test", "cleanupPeriodDays": 0})
    ctx.use(apply, {"maxInlineTokens": 10})

    # Register a mock LLM service with image pricing
    class MockLLM:
        def __init__(self):
            self.image_request_pricing = lambda *args: [{"visualTokens": 50}]
            self.providers = {"test": {}}
            self.active_provider = "test"

    ctx._register_service("llm", MockLLM())

    content = [{"type": "text", "text": "a" * 100}]
    result = await ctx.waterfall(
        "tools/post-execute",
        {
            "name": "test",
            "callId": "c1",
            "agent": {
                "session": {"header": {"id": "session-1"}},
                "provider": "test",
                "model": "test-model",
            },
        },
        {"content": content, "isError": False},
        default_handler=lambda *_a: {"kind": "accept", "content": content},
    )
    assert result["kind"] == "accept"
    assert len(result["content"]) >= 1


@pytest.mark.asyncio
async def test_policy_small_content_passthrough() -> None:
    """Small content passes through unchanged."""
    ctx = Context()
    ctx.use(LocalSpillStore, {"root": "/tmp/spill-test", "cleanupPeriodDays": 0})
    ctx.use(apply, {"maxInlineTokens": 1000})

    content = [{"type": "text", "text": "small"}]
    result = await ctx.waterfall(
        "tools/post-execute",
        {"name": "test", "callId": "c1"},
        {"content": content, "isError": False},
        default_handler=lambda *_a: {"kind": "accept", "content": content},
    )
    assert result["kind"] == "accept"
    assert result["content"] == content


def test_price_block_image_with_pricing() -> None:
    """_price_block prices image blocks when pricing function is available."""
    from magitech_harness.spill.policy import _price_block

    def pricing(images):
        return [{"visualTokens": 50}]

    block = {"type": "image", "attachment": {"attachmentId": "img-1"}}
    result = _price_block(block, pricing)
    assert result == 50


def test_price_block_image_no_pricing() -> None:
    """_price_block returns 0 for image when no pricing available."""
    from magitech_harness.spill.policy import _price_block

    block = {"type": "image", "attachment": {"attachmentId": "img-1"}}
    result = _price_block(block, None)
    assert result == 0


@pytest.mark.asyncio
async def test_policy_llm_none() -> None:
    """apply works without llm service - pricing is None."""
    ctx = Context()
    apply(ctx, {"maxInlineTokens": 10})
    content = [{"type": "text", "text": "a" * 100}]
    result = await ctx.waterfall(
        "tools/post-execute",
        {"name": "test", "callId": "c1"},
        {"content": content, "isError": False},
        default_handler=lambda *_a: {"kind": "accept", "content": content},
    )
    assert result["kind"] == "accept"


@pytest.mark.asyncio
async def test_apply_no_max_inline_tokens() -> None:
    """apply() with no maxInlineTokens is a no-op."""
    ctx = Context()
    apply(ctx, {})
    assert ctx.get("spillStore", None, strict=False) is None


@pytest.mark.asyncio
async def test_policy_non_accept_decision_passthrough() -> None:
    """Non-accept decision passes through unchanged."""
    ctx = Context()
    apply(ctx, {"maxInlineTokens": 10})
    result = await ctx.waterfall(
        "tools/post-execute",
        {"name": "test", "callId": "c1"},
        {"content": [{"type": "text", "text": "text"}], "isError": False},
        default_handler=lambda *_a: {"kind": "deny", "reason": "denied"},
    )
    assert result["kind"] == "deny"


@pytest.mark.asyncio
async def test_policy_non_list_content_passthrough() -> None:
    """Non-list content passes through unchanged."""
    ctx = Context()
    apply(ctx, {"maxInlineTokens": 10})
    result = await ctx.waterfall(
        "tools/post-execute",
        {"name": "test", "callId": "c1"},
        {"content": "not a list", "isError": False},
        default_handler=lambda *_a: {"kind": "accept", "content": "not a list"},
    )
    assert result["kind"] == "accept"


@pytest.mark.asyncio
async def test_policy_parent_no_images_passthrough() -> None:
    """Parent context with no images passes through."""
    ctx = Context()
    apply(ctx, {"maxInlineTokens": 10})
    result = await ctx.waterfall(
        "tools/post-execute",
        {
            "name": "test",
            "callId": "c1",
            "parent": "parent-1",
        },
        {"content": [{"type": "text", "text": "a" * 100}], "isError": False},
        default_handler=lambda *_a: {
            "kind": "accept",
            "content": [{"type": "text", "text": "a" * 100}],
        },
    )
    assert result["kind"] == "accept"


@pytest.mark.asyncio
async def test_spill_store_save_failure() -> None:
    """Spill store save failure keeps original content."""
    ctx = Context()
    # Don't register spillStore - save will fail
    apply(ctx, {"maxInlineTokens": 10})
    content = [{"type": "text", "text": "a" * 100}]
    result = await ctx.waterfall(
        "tools/post-execute",
        {
            "name": "test",
            "callId": "c1",
            "agent": {"session": {"header": {"id": "s1"}}},
        },
        {"content": content, "isError": False},
        default_handler=lambda *_a: {"kind": "accept", "content": content},
    )
    assert result["kind"] == "accept"
    assert result["content"] == content
