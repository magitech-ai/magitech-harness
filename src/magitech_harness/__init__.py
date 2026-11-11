"""Magitech Harness: Composable Agent Runtime by Magitech."""

from __future__ import annotations

__version__ = "0.2.0"

from pathlib import Path
from typing import ClassVar

from anima import Context

from magitech_harness.agent.cancellation import AbortError, AbortSignal
from magitech_harness.agent.loop import AgentLoopService
from magitech_harness.approval.service import (
    ApprovalRequest,
    ApprovalService,
    calculate_argument_digest,
)
from magitech_harness.components.layers import ConfigLayer, LayerResolutionService
from magitech_harness.components.manifest import ComponentManifest
from magitech_harness.components.service import ComponentService
from magitech_harness.llm.service import (
    LLMProvider,
    LLMService,
    MockLLMProvider,
    NoProviderError,
)
from magitech_harness.llm.slug import parse_model_slug
from magitech_harness.llm.types import ModelChunk, ModelResponse, UsageData
from magitech_harness.session.events import (
    EventType,
    SessionEvent,
    ToolCallData,
    ToolResultData,
)
from magitech_harness.session.pairing import is_tool_pairing_balanced
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session, derive_messages
from magitech_harness.shell.service import LocalShellService, ShellResult
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.definition import Tool, define_tool
from magitech_harness.tools.service import ToolService


def create_harness(
    storage_dir: Path | str | None = None,
    *,
    mock_provider: bool = False,
) -> Context:
    """Factory creating a standard Context populated with core Harness services.

    Args:
        storage_dir: Optional base directory to persist session JSONL files.
        mock_provider: If True, mounts an offline MockLLMProvider for testing.

    Returns:
        A configured Context instance.
    """
    ctx = Context()

    base_path = Path(storage_dir) if storage_dir else None

    # Mount core capability seams
    ctx.use(SessionService, base_storage_dir=base_path)
    ctx.use(SystemPromptService)
    ctx.use(ToolService)
    ctx.use(LLMService)
    ctx.use(LocalShellService)
    ctx.use(ApprovalService)
    ctx.use(LayerResolutionService)
    ctx.use(AgentLoopService)
    ctx.use(ComponentService)

    if mock_provider:

        class _MockProviderComponent:
            inject: ClassVar[list[str]] = ["llm"]

            def __init__(self, c: Context) -> None:
                c.llm.register_provider("mock", MockLLMProvider())

        ctx.use(_MockProviderComponent)

    return ctx


__all__ = [
    "AbortError",
    "AbortSignal",
    "AgentLoopService",
    "ApprovalRequest",
    "ApprovalService",
    "ComponentManifest",
    "ComponentService",
    "ConfigLayer",
    "EventType",
    "LLMProvider",
    "LLMService",
    "LayerResolutionService",
    "LocalShellService",
    "MockLLMProvider",
    "ModelChunk",
    "ModelResponse",
    "NoProviderError",
    "Session",
    "SessionEvent",
    "SessionService",
    "ShellResult",
    "SystemPromptService",
    "Tool",
    "ToolCallData",
    "ToolResultData",
    "ToolService",
    "UsageData",
    "calculate_argument_digest",
    "create_harness",
    "define_tool",
    "derive_messages",
    "is_tool_pairing_balanced",
    "parse_model_slug",
]
