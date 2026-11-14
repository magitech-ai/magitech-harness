"""Magitech Harness: Composable Agent Runtime by Magitech."""

from __future__ import annotations

__version__ = "0.2.0"

import asyncio
from pathlib import Path
from typing import ClassVar

from anima import Context, FiberState
from anima.loader import LoaderService, load_profile

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

_BASE_BUNDLE_PATH = Path(__file__).resolve().parent / "profiles" / "base.yaml"


def _check_bundle_entries(loader: LoaderService) -> None:
    """Fail loud if any enabled bundle row did not mount.

    A row that fails to mount never raises through load or flush: the
    failure is visible only on the entry fiber's state, so it is
    checked here. Disabled rows are intentionally unmounted and skipped.
    """
    problems: list[str] = []
    for entry in loader.entries():
        if entry.disabled:
            continue
        fiber = entry.fiber
        if fiber is None:
            problems.append(f"{entry.id}: no fiber mounted")
        elif fiber.state is not FiberState.ACTIVE:
            detail = f": {fiber.error}" if fiber.error is not None else ""
            problems.append(f"{entry.id}: fiber {fiber.state.value}{detail}")
    if problems:
        raise RuntimeError("default bundle failed to mount: " + "; ".join(problems))


def _settle_default_bundle(ctx: Context, loader: LoaderService) -> None:
    """Settle the default bundle rows and fail loud on any row that did not mount.

    With no running event loop the rows settle here, so the returned
    context has every bundle slot. With a running loop the rows settle
    on the caller's flush, following the loader's established
    load/flush contract, and the fiber check does not run yet.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(ctx.flush())
        _check_bundle_entries(loader)


def create_harness(
    storage_dir: Path | str | None = None,
    *,
    mock_provider: bool = False,
) -> Context:
    """Factory creating a standard Context populated with core Harness services.

    The six leaf services (system prompt, tools, LLM, shell, approvals,
    layers) mount from the default bundle file; sessions, the agent
    loop, and components still mount directly. The bundle loads before
    the direct mounts so the agent loop's dependencies are present
    when it mounts.

    Args:
        storage_dir: Optional base directory to persist session JSONL files.
        mock_provider: If True, mounts an offline MockLLMProvider for testing.

    Returns:
        A configured Context instance.
    """
    ctx = Context()

    base_path = Path(storage_dir) if storage_dir else None

    # Mount the default bundle rows first: the six leaf services.
    loader = load_profile(ctx, _BASE_BUNDLE_PATH)
    _settle_default_bundle(ctx, loader)

    # Mount the remaining services directly for now.
    ctx.use(SessionService, base_storage_dir=base_path)
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
