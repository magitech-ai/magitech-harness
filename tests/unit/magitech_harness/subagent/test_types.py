"""Unit tests for subagent types and data structures."""

from pathlib import Path

from magitech_harness.subagent.types import (
    RESEARCH_PRESET_TOOLS,
    SubagentRecord,
    SubagentSpec,
    SubagentState,
)


def test_subagent_spec_defaults() -> None:
    spec = SubagentSpec(role="researcher", prompt="inspect repo")
    assert spec.role == "researcher"
    assert spec.prompt == "inspect repo"
    assert spec.agent_name is None
    assert spec.model == "inherit"
    assert spec.workspace == "inherit"
    assert spec.allowed_tools is None
    assert spec.wait is True


def test_subagent_record_to_dict() -> None:
    rec = SubagentRecord(
        subagent_id="sub-12345",
        role="worker",
        agent_name="agent-1",
        model_id="test-model",
        workspace_path=Path("/tmp/ws"),
        session_id="session-sub-12345",
        session_path=Path("/tmp/ws/sub.jsonl"),
        state=SubagentState.COMPLETED,
        state_detail="Done",
        summary="Task succeeded",
        error=None,
        effective_tools=["read_file"],
    )
    d = rec.to_dict()
    assert d["subagent_id"] == "sub-12345"
    assert d["role"] == "worker"
    assert d["status"] == "completed"
    assert d["summary"] == "Task succeeded"
    assert d["effective_tools"] == ["read_file"]


def test_research_preset_tools() -> None:
    assert "read_file" in RESEARCH_PRESET_TOOLS
    assert "lsp_definition" in RESEARCH_PRESET_TOOLS
    assert "write_file" not in RESEARCH_PRESET_TOOLS
