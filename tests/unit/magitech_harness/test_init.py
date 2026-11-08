"""Unit tests for magitech_harness package exports."""

from anima import Context

import magitech_harness


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
    assert ctx.has("plugins")
    assert not ctx.llm.providers

    # With mock_provider=True
    ctx_mock = magitech_harness.create_harness(mock_provider=True)
    assert "mock" in ctx_mock.llm.providers
