"""Magitech Harness: Composable Agent Runtime by Magitech."""

from __future__ import annotations

from pathlib import Path
from anima import Context
from magitech_harness.agent.loop import AgentLoopService
from magitech_harness.llm.service import LLMService, MockLLMProvider
from magitech_harness.llm.types import ModelResponse
from magitech_harness.session.events import EventType, SessionEvent, ToolCallData, ToolResultData
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session, derive_messages
from magitech_harness.shell.service import LocalShellService
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.definition import Tool, define_tool
from magitech_harness.tools.service import ToolService


def create_harness(storage_dir: Path | str | None = None) -> Context:
    """Factory creating a standard Context populated with all core Harness services."""
    ctx = Context()

    base_path = Path(storage_dir) if storage_dir else None

    # Mount the 6 core capability seams
    ctx.plugin(SessionService, base_storage_dir=base_path)
    ctx.plugin(SystemPromptService)
    ctx.plugin(ToolService)
    ctx.plugin(LLMService)
    ctx.plugin(LocalShellService)
    ctx.plugin(AgentLoopService)

    return ctx


__all__ = [
    "create_harness",
    "SessionService",
    "Session",
    "derive_messages",
    "SessionEvent",
    "EventType",
    "ToolCallData",
    "ToolResultData",
    "SystemPromptService",
    "ToolService",
    "Tool",
    "define_tool",
    "LLMService",
    "MockLLMProvider",
    "ModelResponse",
    "AgentLoopService",
    "LocalShellService",
]
