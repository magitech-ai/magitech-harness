"""Chunk-level acceptance proof for the all-plugin kernel (K6).

Tests only: this file proves the finished kernel shape in one place
instead of scattering the claim across the per-chunk files. It
consolidates the per-chunk acceptance proofs (swapped-row probes,
live-title proof, child-turn proof) and adds the phase-level proofs
the earlier chunks could not state: the full bundle inventory, the
replay on the row boot, the CLI driven end to end, the disposal
proof, and the audit's pre-flush behaviour pinned as it stands.
"""

import re
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import pytest
from anima import Context, FiberState, Service
from anima.jsexpr import dump_yaml, load_yaml
from anima.loader import LoaderService, load_profile

import magitech_harness
from magitech_harness import (
    MockLLMProvider,
    check_bundle_entries,
    create_harness,
    derive_messages,
)
from magitech_harness.agent.loop import AgentLoopService
from magitech_harness.approval.service import ApprovalService
from magitech_harness.cli import main_async
from magitech_harness.components.layers import LayerResolutionService
from magitech_harness.components.service import ComponentService
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


class _MockProviderComponent:
    """Test-only component registering the offline mock provider."""

    inject: ClassVar[list[str]] = ["llm"]

    def __init__(self, ctx: Context) -> None:
        ctx.llm.register_provider("mock", MockLLMProvider())


def _base_bundle_path() -> Path:
    return Path(magitech_harness.__file__).resolve().parent / "profiles" / "base.yaml"


def _base_rows() -> list[dict[str, Any]]:
    return load_yaml(_base_bundle_path().read_text(encoding="utf-8"))


def _write_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    bundle_path = tmp_path / "bundle.yaml"
    with bundle_path.open("w", encoding="utf-8") as handle:
        dump_yaml(rows, handle)
    return bundle_path


def _log_first_message(service: SessionService, session: Any) -> None:
    service.log_event(session, "turn/start", {"turn": 1})
    service.log_event(
        session,
        "user/message",
        {
            "content": [{"type": "text", "text": "Hello"}],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )


def _provider_title_events(session: Any) -> list[Any]:
    return [
        event
        for event in session.events
        if event.type == "session/title"
        and event.payload.get("source", {}).get("kind") == "provider"
    ]


# ---------------------------------------------------------------------------
# (a) Bundle parity proof: the nine services, the owned components, the
# replay, the CLI, the reversed bundle, and the child turn.
# ---------------------------------------------------------------------------


def test_bundle_boot_exposes_nine_services_and_owned_components() -> None:
    """The row boot exposes every service and owned component by slot.

    The nine services the factory used to mount directly, plus the four
    owned components that used to hide inside the sessions service, all
    resolve via ``ctx.<name>`` with the same types and the same hollow
    memory-only state as the per-chunk tests pinned separately.
    """
    ctx = create_harness()
    assert isinstance(ctx.system_prompt, SystemPromptService)
    assert isinstance(ctx.tools, ToolService)
    assert isinstance(ctx.llm, LLMService)
    assert isinstance(ctx.shell, LocalShellService)
    assert isinstance(ctx.approvals, ApprovalService)
    assert isinstance(ctx.layers, LayerResolutionService)
    assert isinstance(ctx.components, ComponentService)
    assert isinstance(ctx.sessions, SessionService)
    assert isinstance(ctx.agent_loop, AgentLoopService)
    assert isinstance(ctx.sessionProjections, SessionProjectionRegistry)
    assert isinstance(ctx.sessionTitle, SessionTitleService)
    assert isinstance(ctx.sessionPersistence, JsonlSessionPersistenceService)
    assert isinstance(ctx.sessionProjectionCache, SessionProjectionCacheService)
    assert ctx.approvals.default_verdict is False
    assert ctx.sessions.base_storage_dir is None
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
    check_bundle_entries(ctx)


def test_create_harness_mounts_nothing_directly() -> None:
    """No direct mount remains in the factory: the bundle is the only source."""
    source = Path(magitech_harness.__file__).read_text(encoding="utf-8")
    start = source.index("def create_harness(")
    body = source[start:]
    assert "ctx.use(" not in body
    assert "mock_provider" not in body


async def test_replay_reproduces_identical_message_sequence(
    tmp_path: Path,
) -> None:
    """A turn written on the row boot replays identically from a fresh boot.

    The recorded turn is written by a bundle-booted harness, drained so
    its provider title is committed, flushed to disk, and read back by
    a second bundle-booted harness on the same directory: the event
    type sequence and the derived message sequence match exactly.
    """
    ctx = create_harness(storage_dir=tmp_path)
    ctx.use(_MockProviderComponent)
    await ctx.flush()
    session = ctx.sessions.create("k6-replay")
    result = await ctx.agent_loop.run_turn(session, "Explain quantum computing simply")
    assert result == "Default mock response"
    await ctx.sessionTitle.drain()
    assert len(_provider_title_events(session)) == 1
    await ctx.sessions.flush()
    first_types = [event.type for event in session.events]
    first_messages = derive_messages(session.events)
    # Release the write lease before the second boot opens the same log.
    assert ctx.sessions.close("k6-replay") is True

    ctx2 = create_harness(storage_dir=tmp_path)
    await ctx2.flush()
    loaded = ctx2.sessions.load("k6-replay")
    assert loaded is not None
    assert [event.type for event in loaded.events] == first_types
    assert derive_messages(loaded.events) == first_messages
    assert ctx2.sessionTitle.get(loaded)["title"] == "Default mock response"


async def test_full_bundle_reversed_still_wires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole bundle reversed still wires: row order carries no semantics.

    Every row moved to the opposite end of the file, the bundle still
    settles, a turn runs, the provider title fires, the cache installs,
    projection reads answer live, and a rename still pins.
    """
    rows = _base_rows()
    rows.reverse()
    monkeypatch.setattr(
        magitech_harness, "_BASE_BUNDLE_PATH", _write_rows(tmp_path, rows)
    )
    ctx = create_harness(tmp_path / "sessions")
    await ctx.flush()
    check_bundle_entries(ctx)
    ctx.use(_MockProviderComponent)
    await ctx.flush()
    assert ctx.sessions.projection_cache is not None
    session = ctx.sessions.create("k6-reversed")
    result = await ctx.agent_loop.run_turn(session, "Explain quantum computing simply")
    assert result == "Default mock response"
    await ctx.sessionTitle.drain()
    titles = _provider_title_events(session)
    assert len(titles) == 1
    assert titles[0].payload["title"] == "Default mock response"
    assert ctx.sessions.read_projections("k6-reversed") is not None
    ctx.sessionTitle.rename(session, "My Reversed Title")
    await ctx.sessionTitle.drain()
    assert ctx.sessionTitle.get(session)["title"] == "My Reversed Title"
    assert ctx.sessionTitle.get(session)["source"] == {"kind": "user"}


async def test_child_harness_from_bundle_runs_a_turn(tmp_path: Path) -> None:
    """A child harness built from the bundle runs a turn of its own.

    The subagent shape at harness level: a second factory-built context
    on its own storage directory runs a turn with the mock provider,
    fires its own provider title, and leaves the parent's sessions
    untouched.
    """
    parent = create_harness(storage_dir=tmp_path / "parent")
    await parent.flush()
    child = create_harness(storage_dir=tmp_path / "child")
    child.use(_MockProviderComponent)
    await child.flush()
    session = child.sessions.create("k6-child")
    result = await child.agent_loop.run_turn(
        session, "Explain quantum computing simply"
    )
    assert result == "Default mock response"
    await child.sessionTitle.drain()
    titles = _provider_title_events(session)
    assert len(titles) == 1
    assert titles[0].payload["title"] == "Default mock response"
    assert parent.sessions.load("k6-child") is None


async def test_cli_commands_end_to_end_on_row_boot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI drives run, session list/show, component list, and repl.

    One storage directory, four commands, one boot shape each: ``run``
    records the turn, ``session list`` names it, ``session show``
    prints it, ``component list`` reports the empty component set, and
    the repl lists the stored session before exiting.
    """
    await main_async(["--storage", str(tmp_path), "run", "Calculate 2+2"])
    run_out = capsys.readouterr().out
    assert "Running prompt: 'Calculate 2+2'" in run_out
    assert "Assistant: Default mock response" in run_out

    await main_async(["--storage", str(tmp_path), "session", "list"])
    list_out = capsys.readouterr().out
    match = re.search(r"  - (\S+)", list_out)
    assert match is not None
    session_id = match.group(1)

    await main_async(["--storage", str(tmp_path), "session", "show", session_id])
    show_out = capsys.readouterr().out
    assert "Calculate 2+2" in show_out

    await main_async(["--storage", str(tmp_path), "component", "list"])
    component_out = capsys.readouterr().out
    assert "Loaded components (0):" in component_out

    inputs = iter(["/session list", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    await main_async(["--storage", str(tmp_path), "repl"])
    repl_out = capsys.readouterr().out
    assert session_id in repl_out
    assert "Goodbye." in repl_out


# ---------------------------------------------------------------------------
# (c) Boot-time slot assertion, and the pre-flush audit pinned as it stands.
# ---------------------------------------------------------------------------


async def test_check_bundle_entries_fails_before_flush_on_loop_path() -> None:
    """The audit fails loud before flush on the loop path when no entries exist.

    Under a running loop the bundle rows have no loader entries until
    the caller's flush, so checking before flush raises RuntimeError
    reporting zero entries rather than passing vacuously. After the flush
    the audit checks all twenty rows.
    """
    ctx = create_harness()
    loader = ctx.get("loader", None, strict=False)
    assert list(loader.entries()) == []
    with pytest.raises(RuntimeError, match="no bundle entries"):
        check_bundle_entries(ctx)
    await ctx.flush()
    assert len(list(loader.entries())) == 20
    check_bundle_entries(ctx)


def test_check_bundle_entries_sync_path_settles_at_load() -> None:
    """With no running loop the rows settle at load, before any flush.

    The sync path is not vacuous: loading the bundle mounts all twenty
    rows ACTIVE at once, the slots are reachable immediately, and the
    audit passes on the settled rows.
    """
    ctx = Context()
    loader = LoaderService(ctx, {"filename": _base_bundle_path()})
    loader.load(_base_bundle_path())
    entries = list(loader.entries())
    assert len(entries) == 20
    assert all(
        entry.fiber is not None and entry.fiber.state is FiberState.ACTIVE
        for entry in entries
    )
    assert ctx.has("tools")
    check_bundle_entries(ctx)


# ---------------------------------------------------------------------------
# (e) Disposal proof: a disabled row cleanly unmounts its slot.
# ---------------------------------------------------------------------------

_SIDE_EFFECTS: list[str] = []
_DISPOSED: list[str] = []


class _ProbeService(Service):
    """Stand-in row service with an observable listener and disposal flag."""

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "probe_slot")

        def _on_probe_event(*_args: object) -> None:
            _SIDE_EFFECTS.append("fired")

        ctx.on("k6/probe-event", _on_probe_event)
        ctx.effect(lambda: _DISPOSED.append("entry-fiber-disposed"))


def _probe_bundle_rows(tmp_path: Path) -> tuple[list[dict[str, Any]], Path]:
    rows = _base_rows()
    rows.append(
        {
            "id": "k6-probe-row",
            "name": "k6_probe5_target:ProbeService",
            "config": {},
        }
    )
    return rows, _write_rows(tmp_path, rows)


async def test_disabling_row_unmounts_slot_and_listeners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling a row removes its slot, silences its listeners, runs disposal.

    The loader-probe conclusion re-proved as a test, on the real bundle
    with a stand-in row appended: by loader disable, and by reloading
    the bundle without the row. Either path leaves the slot gone, the
    listener silent across two fires, the entry fiber DISPOSED, and the
    entry-fiber disposal effect observed.
    """
    mod = types.ModuleType("k6_probe5_target")
    mod.ProbeService = _ProbeService
    monkeypatch.setitem(sys.modules, "k6_probe5_target", mod)
    rows, bundle_path = _probe_bundle_rows(tmp_path)
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)

    # Path A: loader disable.
    _SIDE_EFFECTS.clear()
    _DISPOSED.clear()
    ctx = create_harness()
    await ctx.flush()
    assert ctx.has("probe_slot")
    ctx.emit("k6/probe-event")
    assert _SIDE_EFFECTS == ["fired"]
    loader = ctx.get("loader", None, strict=False)
    entry = loader.resolve("k6-probe-row")
    assert entry.fiber.state is FiberState.ACTIVE
    await loader.disable("k6-probe-row")
    await ctx.flush()
    assert not ctx.has("probe_slot")
    assert entry.fiber.state is FiberState.DISPOSED
    assert _DISPOSED == ["entry-fiber-disposed"]
    ctx.emit("k6/probe-event")
    ctx.emit("k6/probe-event")
    assert _SIDE_EFFECTS == ["fired"]
    check_bundle_entries(ctx)

    # Path B: reload the bundle without the row.
    _SIDE_EFFECTS.clear()
    _DISPOSED.clear()
    ctx2 = create_harness()
    await ctx2.flush()
    assert ctx2.has("probe_slot")
    ctx2.emit("k6/probe-event")
    assert _SIDE_EFFECTS == ["fired"]
    loader2 = ctx2.get("loader", None, strict=False)
    entry2 = loader2.resolve("k6-probe-row")
    loader2.load([row for row in rows if row["id"] != "k6-probe-row"])
    await ctx2.flush()
    assert not ctx2.has("probe_slot")
    assert entry2.fiber.state is FiberState.DISPOSED
    assert _DISPOSED == ["entry-fiber-disposed"]
    ctx2.emit("k6/probe-event")
    ctx2.emit("k6/probe-event")
    assert _SIDE_EFFECTS == ["fired"]
    check_bundle_entries(ctx2)


async def test_disabling_shell_row_unmounts_slot(tmp_path: Path) -> None:
    """Disabling a real row unmounts its slot and the audit still passes.

    The shell row has no dependents, so its removal is clean: the slot
    goes, the other slots stay, and the audit skips the disabled row
    instead of failing on it.
    """
    ctx = create_harness(storage_dir=tmp_path)
    await ctx.flush()
    loader = ctx.get("loader", None, strict=False)
    await loader.disable("shell")
    await ctx.flush()
    assert not ctx.has("shell")
    assert ctx.has("sessions")
    assert ctx.has("tools")
    check_bundle_entries(ctx)


async def test_disabling_policy_row_removes_checkpoint_listeners(
    tmp_path: Path,
) -> None:
    """Disabling the checkpoint-policy row removes all three listeners.

    Observed through the session/flush bus event: with the row active a
    session-bound model stream flushes first; with the row disabled the
    same stream dispatches with no flush while the adapter still runs.
    """
    ctx = create_harness(storage_dir=tmp_path)
    ctx.use(_MockProviderComponent)
    await ctx.flush()
    ctx.llm.set_active_provider("mock")
    flushes: list[str] = []
    ctx.on("session/flush", lambda session: flushes.append(session.session_id))
    session = ctx.sessions.create("k6-policy")
    chunks = [
        chunk
        async for chunk in ctx.llm.stream(messages=[], session_id=session.session_id)
    ]
    assert flushes == ["k6-policy"]
    assert [chunk.text for chunk in chunks] == ["Default mock response"]
    loader = ctx.get("loader", None, strict=False)
    await loader.disable("session-checkpoint-policy")
    await ctx.flush()
    flushes.clear()
    chunks = [
        chunk
        async for chunk in ctx.llm.stream(messages=[], session_id=session.session_id)
    ]
    assert flushes == []
    assert [chunk.text for chunk in chunks] == ["Default mock response"]
    check_bundle_entries(ctx)


# ---------------------------------------------------------------------------
# The three unpinned disposal behaviours, pinned as current behaviour.
# ---------------------------------------------------------------------------


def test_bare_service_with_bad_cache_throttles_does_not_raise() -> None:
    """Bad cache throttles on a bare service do not raise: no cache is built.

    The service builds no cache without the cache row, so throttle
    values in its config are never validated here — with or without a
    registry present. The cache row itself validates its own throttles
    and fails loud on bad ones; that validation lives on the row, not
    on this service.
    """
    bad = {"write_every_events": 0, "write_interval_ms": 0}
    bare = SessionService(Context(), dict(bad))
    assert bare.projection_cache is None
    bare.create("k6-throttle-bare")
    ctx = Context()
    SessionProjectionRegistry(ctx)
    with_registry = SessionService(ctx, dict(bad))
    assert with_registry.projection_cache is None
    with_registry.create("k6-throttle-registry")


async def test_unit_row_unload_keeps_title_via_title_service() -> None:
    """Unloading the title unit rows leaves titling working.

    The title service registers the title and titleInput definitions
    itself, so the registry refcounts them: unloading both unit rows
    drops the refcount but keeps the definitions, and the fallback
    title still lands after a drain.
    """
    ctx = create_harness()
    await ctx.flush()
    loader = ctx.get("loader", None, strict=False)
    await loader.disable("projection-title")
    await loader.disable("projection-title-input")
    await ctx.flush()
    check_bundle_entries(ctx)
    service = ctx.sessions
    session = service.create("k6-unit-unload")
    _log_first_message(service, session)
    await ctx.sessionTitle.drain()
    title = ctx.sessionTitle.get(session)
    assert title is not None
    assert title["title"] == "Hello"
    assert title["source"] == {"kind": "fallback"}
    assert "title" in ctx.sessionProjections.summary_keys()


async def test_title_row_unload_disposes_service_via_effect() -> None:
    """Unloading the title row runs the service's dispose through its effect.

    The held service reports disposed afterwards — listener removal
    alone would not do that — no new title event lands once it is gone,
    and the unit rows keep the title definition registered.
    """
    ctx = create_harness()
    await ctx.flush()
    service = ctx.sessions
    session = service.create("k6-title-unload")
    _log_first_message(service, session)
    await ctx.sessionTitle.drain()
    assert ctx.sessionTitle.get(session)["title"] == "Hello"
    title_service = ctx.sessionTitle
    before = [event for event in session.events if event.type == "session/title"]
    loader = ctx.get("loader", None, strict=False)
    await loader.disable("session-title")
    await ctx.flush()
    assert not ctx.has("sessionTitle")
    # The provider row injects the title service, so it can never
    # activate once its dependency is gone: the audit names it PENDING
    # instead of passing. That cascade is the behaviour as it stands.
    with pytest.raises(RuntimeError, match="session-title-first-prompt"):
        check_bundle_entries(ctx)
    with pytest.raises(RuntimeError, match="session-title service disposed"):
        title_service.rename(session, "New name")
    service.log_event(
        session,
        "user/message",
        {
            "content": [{"type": "text", "text": "Second message"}],
            "source": {"kind": "user"},
        },
        surface_op="append",
    )
    after = [event for event in session.events if event.type == "session/title"]
    assert after == before
    assert "title" in ctx.sessionProjections.summary_keys()


# ---------------------------------------------------------------------------
# Direct-mount-wins: the loader conclusion the mixed transition relied on.
# ---------------------------------------------------------------------------


class _DirectService(Service):
    """Stand-in service mounted directly, owning its slot."""

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "probe_slot")
        self.marker = "direct"


class _RowService(Service):
    """Stand-in service mounted as a row, providing the same slot."""

    def __init__(self, ctx: Context, config: dict | None = None) -> None:
        super().__init__(ctx, "probe_slot")
        self.marker = "row"


async def test_direct_mount_wins_over_bundle_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directly mounted service owns its slot; the row stays inert on root.

    The row still runs on its own entry context, and removing the
    direct mount lets the mirror apply: the slot then resolves to the
    row instance.
    """
    direct_mod = types.ModuleType("k6_probe3_direct")
    direct_mod.DirectService = _DirectService
    monkeypatch.setitem(sys.modules, "k6_probe3_direct", direct_mod)
    row_mod = types.ModuleType("k6_probe3_row")
    row_mod.RowService = _RowService
    monkeypatch.setitem(sys.modules, "k6_probe3_row", row_mod)
    ctx = Context()
    direct_fiber = ctx.use(_DirectService)
    loader = load_profile(
        ctx, [{"id": "probe-row", "name": "k6_probe3_row:RowService"}]
    )
    await ctx.flush()
    assert ctx.probe_slot.marker == "direct"
    entry = loader.resolve("probe-row")
    assert entry.fiber.state is FiberState.ACTIVE
    assert entry.ctx.probe_slot.marker == "row"
    await direct_fiber.dispose()
    await ctx.flush()
    assert ctx.probe_slot.marker == "row"
