"""Final branch probes for the relocated subagent tool surface."""

from __future__ import annotations

from typing import Any

import pytest
from animas import Context

from magitech_harness import create_harness
from magitech_harness.llm.service import LLMService
from magitech_harness.subagent.service import SubagentService
from magitech_harness.tools.service import ToolsService


class _Agent:
    def __init__(self, session: Any) -> None:
        self.id = "root"
        self.session = session
        self.inbox = type("Inbox", (), {"has_pending": True})()


@pytest.mark.asyncio
async def test_list_subagent_models_without_llm_rows(tmp_path: Any) -> None:
    ctx = create_harness(storage_dir=tmp_path / "empty-models")
    await ctx.flush()
    session = ctx.sessions.create("empty-models-session")

    class _EmptyLlm:
        def list_providers(self) -> list[dict[str, str]]:
            return [
                {"id": "p"},
                {"provider": "p2", "model": "m"},
                {"provider": "p3"},
                {"provider": "p4", "model": 7},
            ]

    ctx.llm.list_providers = _EmptyLlm().list_providers  # type: ignore[method-assign]
    batch = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "list_subagent_models",
                    "name": "list_subagent_models",
                    "arguments": {},
                }
            ],
            "signal": __import__(
                "magitech_harness.agent.cancellation", fromlist=["AbortSignal"]
            ).AbortSignal(),
            "agent": _Agent(session),
        }
    )
    value = batch["results"][0]["result"]["value"]
    assert value == {
        "models": [
            {"provider": "p", "model": "p"},
            {"provider": "p2", "model": "m"},
        ]
    }


@pytest.mark.asyncio
async def test_list_subagent_models_without_llm_service() -> None:
    ctx = Context()
    ToolsService(ctx)
    SubagentService(ctx)
    from magitech_harness.tools.native_executor import apply as apply_native

    apply_native(ctx)
    apply_control = __import__(
        "magitech_harness.subagent.tools", fromlist=["apply_control"]
    ).apply_control
    apply_control(ctx)
    from magitech_harness.agent.cancellation import AbortSignal

    batch = await ctx.tools.execute(
        {
            "calls": [
                {
                    "type": "tool-call",
                    "id": "list_subagent_models",
                    "name": "list_subagent_models",
                    "arguments": {},
                }
            ],
            "signal": AbortSignal(),
            "agent": _Agent(None),
        }
    )
    assert batch["results"][0]["result"]["value"] == {"models": []}


def test_log_event_with_inactive_sessions() -> None:
    ctx = Context()
    LLMService(ctx)
    service = SubagentService(ctx)
    fake = type("Sessions", (), {"active_session": None})()
    ctx.reflect.provide("sessions", fake)
    service._log_event("subagent/descriptor", {})
