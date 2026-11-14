"""Unit tests for the default bundle file and its loading in create_harness."""

import sys
import types
from pathlib import Path

import pytest
from anima import Context, Service
from anima.jsexpr import load_yaml

import magitech_harness
from magitech_harness import create_harness
from magitech_harness.approval.service import ApprovalService
from magitech_harness.components.layers import LayerResolutionService
from magitech_harness.llm.service import LLMService
from magitech_harness.shell.service import LocalShellService
from magitech_harness.system_prompt.service import SystemPromptService
from magitech_harness.tools.service import ToolService

EXPECTED_ROWS = [
    {
        "id": "system-prompt",
        "name": "magitech_harness.system_prompt.service:SystemPromptService",
        "config": {},
    },
    {
        "id": "tools",
        "name": "magitech_harness.tools.service:ToolService",
        "config": {},
    },
    {
        "id": "llm",
        "name": "magitech_harness.llm.service:LLMService",
        "config": {},
    },
    {
        "id": "shell",
        "name": "magitech_harness.shell.service:LocalShellService",
        "config": {},
    },
    {
        "id": "approvals",
        "name": "magitech_harness.approval.service:ApprovalService",
        "config": {"default_verdict": False},
    },
    {
        "id": "layers",
        "name": "magitech_harness.components.layers:LayerResolutionService",
        "config": {},
    },
]


def test_base_bundle_rows() -> None:
    """Verifies the default bundle file carries exactly the six leaf rows."""
    bundle_path = (
        Path(magitech_harness.__file__).resolve().parent / "profiles" / "base.yaml"
    )
    with open(bundle_path, encoding="utf-8") as f:
        rows = load_yaml(f)
    assert rows == EXPECTED_ROWS


def test_bundle_slots_are_row_services() -> None:
    """Verifies the six bundle slots resolve to the row services."""
    ctx = create_harness()
    assert isinstance(ctx.system_prompt, SystemPromptService)
    assert isinstance(ctx.tools, ToolService)
    assert isinstance(ctx.llm, LLMService)
    assert isinstance(ctx.shell, LocalShellService)
    assert isinstance(ctx.approvals, ApprovalService)
    assert isinstance(ctx.layers, LayerResolutionService)
    assert ctx.approvals.default_verdict is False
    assert not ctx.llm.providers


@pytest.mark.asyncio
async def test_bundle_slots_settle_on_flush_inside_running_loop() -> None:
    """Verifies bundle rows are pending until the caller's flush on a running loop."""
    ctx = create_harness()
    assert not ctx.has("tools")
    assert not ctx.has("llm")
    await ctx.flush()
    assert ctx.has("tools")
    assert ctx.has("llm")
    assert isinstance(ctx.tools, ToolService)
    assert isinstance(ctx.llm, LLMService)


def _write_bundle(tmp_path: Path, text: str) -> Path:
    bundle_path = tmp_path / "bundle.yaml"
    bundle_path.write_text(text, encoding="utf-8")
    return bundle_path


def test_create_harness_fails_loud_on_unresolvable_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies a bundle row naming an unresolvable target fails loud."""
    bundle_path = _write_bundle(
        tmp_path,
        "- id: bad-row\n  name: nonexistent.module:Nope\n  config: {}\n",
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    with pytest.raises(RuntimeError, match="bad-row"):
        create_harness()


def test_create_harness_fails_loud_on_failing_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies a bundle row whose service raises at init fails loud."""

    class RaisingService(Service):
        """Service that raises during construction."""

        def __init__(self, ctx: Context, config: dict | None = None) -> None:
            raise ValueError("boom-from-init")

    mod = types.ModuleType("k1_test_failing_services")
    mod.RaisingService = RaisingService
    monkeypatch.setitem(sys.modules, "k1_test_failing_services", mod)
    bundle_path = _write_bundle(
        tmp_path,
        "- id: raising-row\n"
        "  name: k1_test_failing_services:RaisingService\n"
        "  config: {}\n",
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    with pytest.raises(RuntimeError, match="raising-row"):
        create_harness()


def test_create_harness_ignores_disabled_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies a disabled bundle row is skipped, not treated as a failure."""
    rows_text = "".join(
        f"- id: {row['id']}\n  name: {row['name']}\n  config: {{}}\n"
        for row in EXPECTED_ROWS
    )
    bundle_path = _write_bundle(
        tmp_path,
        rows_text + "- id: off-row\n"
        "  name: nonexistent.module:Nope\n"
        "  disabled: true\n",
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    ctx = create_harness()
    assert ctx.has("tools")
    assert ctx.has("llm")
    assert not ctx.has("off-row")
