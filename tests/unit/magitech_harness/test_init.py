"""Unit tests for magitech_harness package exports."""

from typing import ClassVar

from anima import Context

import magitech_harness
from magitech_harness import MockLLMProvider


class _MockProviderComponent:
    """Test-only component registering the offline mock provider."""

    inject: ClassVar[list[str]] = ["llm"]

    def __init__(self, ctx: Context) -> None:
        ctx.llm.register_provider("mock", MockLLMProvider())


def test_package_exports() -> None:
    """Verifies that all symbols listed in __all__ are exported correctly."""
    for symbol in magitech_harness.__all__:
        assert hasattr(magitech_harness, symbol), (
            f"Symbol '{symbol}' not found in __all__"
        )


def test_create_harness_default() -> None:
    """Verifies that create_harness boots an Anima Context with all core services."""
    ctx = magitech_harness.create_harness()
    assert isinstance(ctx, Context)
    assert ctx.has("sessions")
    assert ctx.has("system_prompt")
    assert ctx.has("tools")
    assert ctx.has("llm")
    assert ctx.has("shell")
    assert ctx.has("approvals")
    assert ctx.has("layers")
    assert ctx.has("agent_loop")
    assert ctx.has("components")
    assert not ctx.llm.providers

    # With the mock provider mounted as a test-only component
    ctx_mock = magitech_harness.create_harness()
    ctx_mock.use(_MockProviderComponent)
    assert "mock" in ctx_mock.llm.providers
