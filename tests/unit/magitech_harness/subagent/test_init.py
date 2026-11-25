"""Unit tests for subagent package exports."""

import magitech_harness.subagent as subagent_pkg


def test_init_exports() -> None:
    assert hasattr(subagent_pkg, "SubagentManager")
    assert hasattr(subagent_pkg, "SubagentRecord")
    assert hasattr(subagent_pkg, "SubagentSpec")
    assert hasattr(subagent_pkg, "SubagentState")
    assert hasattr(subagent_pkg, "RESEARCH_PRESET_TOOLS")
    assert hasattr(subagent_pkg, "apply")
