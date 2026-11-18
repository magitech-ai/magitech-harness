"""Phase 8 C4 conformance: single driver and single tool surface (findings #1, #5, #7).

These are live structural probes: they read the shipped profile and import the
package the way the CLI does, so a reintroduced legacy driver, a second tool
executor, or a second loop surface fails the suite rather than silently
coexisting. They complement the behavioral E2E tests that drive the unified
loop but never asserted consolidation.
"""

from __future__ import annotations

from pathlib import Path

from anima.jsexpr import load_yaml

import magitech_harness


def _base_rows() -> list[dict[str, object]]:
    path = Path(magitech_harness.__file__).resolve().parent / "profiles" / "base.yaml"
    with open(path, encoding="utf-8") as stream:
        rows = load_yaml(stream)
    assert isinstance(rows, list)
    return rows


def test_single_agent_driver_row_and_no_legacy_module() -> None:
    """Finding #1/#7: exactly one loop row is mounted and the legacy module is gone."""
    rows = _base_rows()
    loop_rows = [r for r in rows if str(r.get("id", "")).startswith("agent-loop")]
    assert loop_rows == [
        {
            "id": "agent-loop",
            "name": "magitech_harness.agent.agent_loop:AgentLoop",
            "config": {"agents": []},
        }
    ]
    legacy = Path(magitech_harness.__file__).resolve().parent / "agent" / "loop.py"
    assert not legacy.exists(), "legacy agent driver reintroduced"

    package = Path(magitech_harness.__file__).resolve().parent
    assert not (package / "agent" / "loop.py").exists()
    # The unified driver is importable and is the only AgentLoop surface.
    from magitech_harness.agent.agent_loop import AgentLoop

    assert AgentLoop.__module__ == "magitech_harness.agent.agent_loop"


def test_single_tool_surface_and_one_executor() -> None:
    """Finding #5: one ToolsService and one native executor row, no second surface."""
    rows = _base_rows()
    tools_rows = [r for r in rows if r.get("id") == "tools"]
    assert len(tools_rows) == 1
    assert tools_rows[0]["name"] == "magitech_harness.tools.service:ToolService"

    executor_rows = [r for r in rows if "executor" in str(r.get("id", ""))]
    assert [r.get("name") for r in executor_rows] == [
        "magitech_harness.tools.native_executor:apply"
    ]
