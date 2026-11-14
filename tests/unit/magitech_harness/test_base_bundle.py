"""Unit tests for the default bundle file and its loading in create_harness."""

import sys
import types
from pathlib import Path

import pytest
from anima import Context, Service
from anima.jsexpr import dump_yaml, load_yaml

import magitech_harness
from magitech_harness import EventType, check_bundle_entries, create_harness
from magitech_harness.agent.loop import AgentLoopService
from magitech_harness.approval.service import ApprovalService
from magitech_harness.components.layers import LayerResolutionService
from magitech_harness.llm.service import LLMService
from magitech_harness.session.persistence_jsonl import (
    JsonlSessionPersistenceService,
)
from magitech_harness.session.projection_cache import (
    SessionProjectionCacheService,
)
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService
from magitech_harness.session.title import SessionTitleService
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
    {
        "id": "session-projections",
        "name": "magitech_harness.session.projection_registry:SessionProjectionRegistry",
        "config": {},
    },
    {
        "id": "sessions",
        "name": "magitech_harness.session.service:SessionService",
        "config": {
            "base_storage_dir": "${MAGITECH_SESSIONS_DIR}",
        },
    },
    {
        "id": "session-persistence-jsonl",
        "name": "magitech_harness.session.persistence_jsonl:JsonlSessionPersistenceService",
        "config": {
            "root": "${MAGITECH_SESSIONS_DIR}",
            "live_batch_delay": 0.2,
        },
    },
    {
        "id": "projection-title",
        "name": "magitech_harness.session.projections:apply_title_projection",
        "config": {},
    },
    {
        "id": "projection-title-input",
        "name": "magitech_harness.session.projections:apply_title_input_projection",
        "config": {},
    },
    {
        "id": "projection-session-stats",
        "name": "magitech_harness.session.projections:apply_session_stats_projection",
        "config": {},
    },
    {
        "id": "projection-turn-outline",
        "name": "magitech_harness.session.projections:apply_turn_outline_projection",
        "config": {},
    },
    {
        "id": "projection-working-directory",
        "name": "magitech_harness.session.projections:apply_working_directory_projection",
        "config": {},
    },
    {
        "id": "session-title",
        "name": "magitech_harness.session.title:SessionTitleService",
        "config": {
            "fallback_max_words": 5,
            "fallback_max_bytes": 40,
            "max_title_bytes": 80,
        },
    },
    {
        "id": "session-checkpoint-policy",
        "name": "magitech_harness.session.checkpoint_policy:apply",
        "config": {},
    },
    {
        "id": "session-projection-cache",
        "name": "magitech_harness.session.projection_cache:SessionProjectionCacheService",
        "config": {
            "root": "${MAGITECH_SESSIONS_DIR}",
            "write_every_events": 200,
            "write_interval_ms": 5000,
        },
    },
    {
        "id": "agent-loop",
        "name": "magitech_harness.agent.loop:AgentLoopService",
        "config": {},
    },
]


def test_base_bundle_rows() -> None:
    """Verifies the default bundle file carries exactly the eighteen rows."""
    bundle_path = (
        Path(magitech_harness.__file__).resolve().parent / "profiles" / "base.yaml"
    )
    with open(bundle_path, encoding="utf-8") as f:
        rows = load_yaml(f)
    assert rows == EXPECTED_ROWS


def test_bundle_slots_are_row_services() -> None:
    """Verifies the eighteen bundle slots resolve to the row services."""
    ctx = create_harness()
    assert isinstance(ctx.system_prompt, SystemPromptService)
    assert isinstance(ctx.tools, ToolService)
    assert isinstance(ctx.llm, LLMService)
    assert isinstance(ctx.shell, LocalShellService)
    assert isinstance(ctx.approvals, ApprovalService)
    assert isinstance(ctx.layers, LayerResolutionService)
    assert isinstance(ctx.sessionProjections, SessionProjectionRegistry)
    assert isinstance(ctx.sessions, SessionService)
    assert isinstance(ctx.sessionTitle, SessionTitleService)
    assert isinstance(ctx.agent_loop, AgentLoopService)
    assert isinstance(ctx.sessionPersistence, JsonlSessionPersistenceService)
    assert isinstance(ctx.sessionProjectionCache, SessionProjectionCacheService)
    assert ctx.approvals.default_verdict is False
    assert ctx.sessions.base_storage_dir is None
    # Memory-only boot activates both rows hollow: no backend, no cache.
    persistence = ctx.get("sessionPersistence", None, strict=False)
    assert persistence.backend is None
    assert persistence.active is False
    cache = ctx.get("sessionProjectionCache", None, strict=False)
    assert cache.cache is None
    assert cache.active is False
    assert ctx.sessions.projection_cache is None
    assert not ctx.llm.providers
    assert ctx.sessionProjections.summary_keys() == frozenset(
        {"title", "workingDirectory"}
    )


@pytest.mark.asyncio
async def test_bundle_slots_settle_on_flush_inside_running_loop() -> None:
    """Verifies bundle rows are pending until the caller's flush on a running loop."""
    ctx = create_harness()
    assert not ctx.has("tools")
    assert not ctx.has("llm")
    assert not ctx.has("sessions")
    assert not ctx.has("sessionProjections")
    assert not ctx.has("sessionTitle")
    assert not ctx.has("sessionPersistence")
    assert not ctx.has("sessionProjectionCache")
    assert not ctx.has("agent_loop")
    await ctx.flush()
    assert ctx.has("tools")
    assert ctx.has("llm")
    assert ctx.has("sessions")
    assert ctx.has("sessionProjections")
    assert ctx.has("sessionTitle")
    assert ctx.has("sessionPersistence")
    assert ctx.has("sessionProjectionCache")
    assert ctx.has("agent_loop")
    assert isinstance(ctx.tools, ToolService)
    assert isinstance(ctx.llm, LLMService)
    assert isinstance(ctx.sessions, SessionService)
    assert isinstance(ctx.sessionProjections, SessionProjectionRegistry)
    assert isinstance(ctx.sessionTitle, SessionTitleService)
    assert isinstance(ctx.agent_loop, AgentLoopService)


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
    rows = [dict(row) for row in EXPECTED_ROWS]
    rows.append(
        {
            "id": "off-row",
            "name": "nonexistent.module:Nope",
            "config": {},
            "disabled": True,
        }
    )
    bundle_path = tmp_path / "bundle.yaml"
    with open(bundle_path, "w", encoding="utf-8") as f:
        dump_yaml(rows, f)
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    ctx = create_harness()
    assert ctx.has("tools")
    assert ctx.has("llm")
    assert not ctx.has("off-row")


def test_prepare_runs_before_row_activation() -> None:
    """Verifies the prepare hook runs before any row activates."""
    observed: dict[str, bool] = {}
    seen_ctx: list[Context] = []

    def prepare(ctx: Context) -> None:
        seen_ctx.append(ctx)
        observed["tools"] = ctx.has("tools")
        observed["sessions"] = ctx.has("sessions")

    ctx = create_harness(prepare=prepare)
    assert len(seen_ctx) == 1
    assert seen_ctx[0] is ctx
    assert observed == {"tools": False, "sessions": False}
    assert ctx.has("tools")
    assert ctx.has("sessions")


@pytest.mark.asyncio
async def test_prepare_runs_before_row_activation_on_running_loop() -> None:
    """Verifies prepare runs before activation on the running-loop path too."""
    observed: dict[str, bool] = {}

    def prepare(ctx: Context) -> None:
        observed["tools"] = ctx.has("tools")
        observed["sessions"] = ctx.has("sessions")

    ctx = create_harness(prepare=prepare)
    assert observed == {"tools": False, "sessions": False}
    await ctx.flush()
    assert ctx.has("tools")
    assert ctx.has("sessions")


def test_prepare_error_propagates() -> None:
    """Verifies an error in the prepare hook is not swallowed."""

    def prepare(ctx: Context) -> None:
        raise ValueError("host-setup-failed")

    with pytest.raises(ValueError, match="host-setup-failed"):
        create_harness(prepare=prepare)


def test_check_bundle_entries_raises_without_loader() -> None:
    """Verifies the audit fails loud on a context with no bundle loader."""
    with pytest.raises(TypeError, match="no loader"):
        check_bundle_entries(Context())


@pytest.mark.asyncio
async def test_check_bundle_entries_passes_after_flush() -> None:
    """Verifies the audit passes on a settled bundle under a running loop."""
    ctx = create_harness()
    await ctx.flush()
    check_bundle_entries(ctx)


@pytest.mark.asyncio
async def test_check_bundle_entries_fails_loud_on_unresolvable_row_after_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies the audit fails loud on an unresolvable row on the loop path."""
    bundle_path = _write_bundle(
        tmp_path,
        "- id: bad-row\n  name: nonexistent.module:Nope\n  config: {}\n",
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    ctx = create_harness()
    await ctx.flush()
    with pytest.raises(RuntimeError, match="bad-row"):
        check_bundle_entries(ctx)


@pytest.mark.asyncio
async def test_check_bundle_entries_fails_loud_on_failing_row_after_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies the audit fails loud on a row raising at init on the loop path."""

    class RaisingService(Service):
        """Service that raises during construction."""

        def __init__(self, ctx: Context, config: dict | None = None) -> None:
            raise ValueError("boom-from-init")

    mod = types.ModuleType("k1_test_failing_services_loop")
    mod.RaisingService = RaisingService
    monkeypatch.setitem(sys.modules, "k1_test_failing_services_loop", mod)
    bundle_path = _write_bundle(
        tmp_path,
        "- id: raising-row\n"
        "  name: k1_test_failing_services_loop:RaisingService\n"
        "  config: {}\n",
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    ctx = create_harness()
    await ctx.flush()
    with pytest.raises(RuntimeError, match="raising-row"):
        check_bundle_entries(ctx)


@pytest.mark.asyncio
async def test_sessions_row_receives_storage_dir(tmp_path: Path) -> None:
    """Verifies create_harness(storage_dir=...) reaches the sessions row."""
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    assert ctx.sessions.base_storage_dir == tmp_path
    persistence = ctx.get("sessionPersistence", None, strict=False)
    assert persistence.backend is not None
    assert persistence.backend.root == tmp_path
    cache = ctx.get("sessionProjectionCache", None, strict=False)
    assert cache.cache is not None
    assert ctx.sessions.projection_cache is cache.cache
    session = ctx.sessions.create("k2_storage")
    ctx.sessions.log_event(session, EventType.USER_MESSAGE, {"text": "hi"})
    await ctx.sessions.flush()
    location = ctx.sessions.location(session.session_id)
    assert location is not None
    assert location.exists()
    assert location.is_relative_to(tmp_path)


@pytest.mark.asyncio
async def test_storage_dir_none_ignores_ambient_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies storage_dir=None stays memory-only despite an ambient value."""
    monkeypatch.setenv("MAGITECH_SESSIONS_DIR", str(tmp_path))
    ctx = create_harness()
    await ctx.flush()
    assert ctx.sessions.base_storage_dir is None
    session = ctx.sessions.create("k2_memory_only")
    ctx.sessions.log_event(session, EventType.USER_MESSAGE, {"text": "hi"})
    await ctx.sessions.flush()
    assert ctx.sessions.location(session.session_id) is None
    assert list(tmp_path.iterdir()) == []


def test_create_harness_fails_loud_on_bad_sessions_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies a sessions row naming an unresolvable target fails loud."""
    bundle_path = _write_bundle(
        tmp_path,
        "- id: sessions\n  name: nonexistent.module:Nope\n  config: {}\n",
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    with pytest.raises(RuntimeError, match="sessions"):
        create_harness()


@pytest.mark.asyncio
async def test_check_bundle_entries_fails_loud_on_bad_sessions_row_after_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies the audit fails loud on a bad sessions row on the loop path."""
    bundle_path = _write_bundle(
        tmp_path,
        "- id: sessions\n  name: nonexistent.module:Nope\n  config: {}\n",
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    ctx = create_harness()
    await ctx.flush()
    with pytest.raises(RuntimeError, match="sessions"):
        check_bundle_entries(ctx)
