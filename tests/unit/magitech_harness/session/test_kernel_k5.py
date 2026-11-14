"""Chunk-level tests for the title-provider + components rows (K5).

The first-prompt title provider and the component service mount as
bundle rows now; ``create_harness`` mounts nothing directly, and the
mock provider is a test-only component mounted via ``ctx.use``. These
tests pin the chunk's shape: the provider row registers into the title
service and fires automatic model titles through real turns, a bad
provider config or a missing dependency fails loud at boot, the rows
do not depend on bundle order, and disabling the provider row restores
the deterministic fallback.
"""

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from anima import Context
from anima.jsexpr import dump_yaml, load_yaml

import magitech_harness
from magitech_harness import MockLLMProvider, check_bundle_entries, create_harness
from magitech_harness.llm.service import LLMService
from magitech_harness.session.events import EventType
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService
from magitech_harness.session.store import Session
from magitech_harness.session.title import SessionTitleService
from magitech_harness.session.title_first_prompt import (
    apply as apply_title_first_prompt,
)

PROVIDER_CONFIG = {
    "target_words": 5,
    "target_cjk_characters": 10,
    "max_input_bytes": 4096,
    "max_output_tokens": 4096,
    "timeout_ms": 60000,
}

_TITLE_CONFIG = {
    "fallback_max_words": 5,
    "fallback_max_bytes": 40,
    "max_title_bytes": 80,
}


class _MockProviderComponent:
    """Test-only component registering the offline mock provider."""

    inject: ClassVar[list[str]] = ["llm"]

    def __init__(self, ctx: Context) -> None:
        ctx.llm.register_provider("mock", MockLLMProvider())


def _manual_ctx() -> Context:
    """A bare context with the title stack and a mock model, no bundle."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    SessionService(ctx)
    SessionTitleService(ctx, dict(_TITLE_CONFIG))
    llm = LLMService(ctx)
    llm.register_provider("mock", MockLLMProvider())
    return ctx


def _log_user(service: SessionService, session: Session, text: str) -> None:
    service.log_event(
        session,
        EventType.USER_MESSAGE,
        {
            "content": [{"type": "text", "text": text}],
            "source": {"kind": "user"},
        },
    )


def _log_header(service: SessionService, session: Session) -> None:
    service.log_event(
        session,
        EventType.REQUEST_HEADER,
        {
            "header": {"config": {"provider": "mock", "model": "mock"}},
            "reason": "initial",
        },
    )


def _write_bundle(tmp_path: Path, text: str) -> Path:
    bundle_path = tmp_path / "bundle.yaml"
    bundle_path.write_text(text, encoding="utf-8")
    return bundle_path


def _default_rows() -> list[dict[str, Any]]:
    bundle_path = (
        Path(magitech_harness.__file__).resolve().parent / "profiles" / "base.yaml"
    )
    return load_yaml(bundle_path.read_text(encoding="utf-8"))


def _write_rows(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    bundle_path = tmp_path / "bundle.yaml"
    with bundle_path.open("w", encoding="utf-8") as handle:
        dump_yaml(rows, handle)
    return bundle_path


def test_apply_requires_session_title() -> None:
    """The provider row fails loud on a context with no title service."""
    with pytest.raises(
        RuntimeError,
        match="session-title-first-prompt requires the sessionTitle service",
    ):
        apply_title_first_prompt(Context(), dict(PROVIDER_CONFIG))


def test_apply_requires_llm() -> None:
    """The provider row fails loud on a context with no model service."""
    ctx = Context()
    SessionProjectionRegistry(ctx)
    SessionService(ctx)
    SessionTitleService(ctx, dict(_TITLE_CONFIG))
    with pytest.raises(
        RuntimeError, match="session-title-first-prompt requires the llm service"
    ):
        apply_title_first_prompt(ctx, dict(PROVIDER_CONFIG))


async def test_apply_registers_provider() -> None:
    """The row's apply registers the provider, which then titles from the model."""
    ctx = _manual_ctx()
    dispose = apply_title_first_prompt(ctx, dict(PROVIDER_CONFIG))
    service = ctx.sessions
    title = ctx.sessionTitle
    session = service.create("k5-apply")
    _log_user(service, session, "Explain quantum computing simply")
    _log_header(service, session)
    await title.drain()
    snapshot = title.get(session)
    assert snapshot is not None
    assert snapshot["title"] == "Default mock response"
    assert snapshot["source"] == {
        "kind": "provider",
        "provider": "session-title-first-prompt",
        "model": {"provider": "mock", "model": "mock"},
    }
    request_events = [
        event for event in session.events if event.type == "session/title-llm-request"
    ]
    assert len(request_events) == 1
    assert request_events[0].payload["titleProvider"] == "session-title-first-prompt"
    dispose()
    assert title.get(session)["title"] == "Default mock response"


_TITLE_PROVIDER_BUNDLE = (
    "- id: session-projections\n"
    "  name: magitech_harness.session.projection_registry:SessionProjectionRegistry\n"
    "  config: {}\n"
    "- id: sessions\n"
    "  name: magitech_harness.session.service:SessionService\n"
    "  config: {}\n"
    "- id: llm\n"
    "  name: magitech_harness.llm.service:LLMService\n"
    "  config: {}\n"
    "- id: session-title\n"
    "  name: magitech_harness.session.title:SessionTitleService\n"
    "  config: {title}\n"
    "- id: session-title-first-prompt\n"
    "  name: magitech_harness.session.title_first_prompt:apply\n"
    "  config: {provider}\n"
)


@pytest.mark.parametrize(
    "provider_config",
    [
        "{}",
        (
            '{"target_words": 0, "target_cjk_characters": 10, "max_input_bytes": 4096,'
            ' "max_output_tokens": 4096, "timeout_ms": 60000}'
        ),
        (
            '{"target_words": 5, "target_cjk_characters": 10, "max_input_bytes": 4096,'
            ' "max_output_tokens": 4096, "timeout_ms": 60000, "surprise": 1}'
        ),
    ],
)
def test_title_provider_row_config_fails_loud_at_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_config: str
) -> None:
    """A missing or invalid provider-row config fails loud, naming the row."""
    bundle_path = _write_bundle(
        tmp_path,
        _TITLE_PROVIDER_BUNDLE.replace(
            "{title}",
            '{"fallback_max_words": 5, "fallback_max_bytes": 40,'
            ' "max_title_bytes": 80}',
        ).replace("{provider}", provider_config),
    )
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    with pytest.raises(RuntimeError, match="session-title-first-prompt"):
        create_harness()


def test_title_provider_row_without_title_service_stays_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the title service the provider row never activates, loudly.

    The row declares the title service as a required injection, so a
    bundle without it leaves the fiber pending and the boot audit
    names the row instead of silently running without a provider.
    """
    rows = [row for row in _default_rows() if row["id"] != "session-title"]
    assert any(row["id"] == "session-title-first-prompt" for row in rows)
    monkeypatch.setattr(
        magitech_harness, "_BASE_BUNDLE_PATH", _write_rows(tmp_path, rows)
    )
    with pytest.raises(RuntimeError, match="session-title-first-prompt"):
        create_harness()


def test_components_row_bad_target_fails_loud_at_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A components row naming an unresolvable target fails loud."""
    rows = _default_rows()
    for row in rows:
        if row["id"] == "components":
            row["name"] = "nonexistent.module:Nope"
    monkeypatch.setattr(
        magitech_harness, "_BASE_BUNDLE_PATH", _write_rows(tmp_path, rows)
    )
    with pytest.raises(RuntimeError, match="components"):
        create_harness()


def test_create_harness_mounts_nothing_directly() -> None:
    """No direct mount remains in the factory: the bundle is the only source."""
    source = Path(magitech_harness.__file__).read_text(encoding="utf-8")
    start = source.index("def create_harness(")
    body = source[start:]
    assert "ctx.use(" not in body
    assert "mock_provider" not in body


def _provider_title_events(session: Session) -> list[Any]:
    return [
        event
        for event in session.events
        if event.type == "session/title"
        and event.payload.get("source", {}).get("kind") == "provider"
    ]


async def test_automatic_title_fires_through_run_turn() -> None:
    """A run_turn commits the model title from the wired provider row."""
    ctx = create_harness()
    ctx.use(_MockProviderComponent)
    await ctx.flush()
    session = ctx.sessions.create("k5-run-turn")
    result = await ctx.agent_loop.run_turn(session, "Explain quantum computing simply")
    assert result == "Default mock response"
    await ctx.sessionTitle.drain()
    titles = _provider_title_events(session)
    assert len(titles) == 1
    assert titles[0].payload["title"] == "Default mock response"
    assert titles[0].payload["source"] == {
        "kind": "provider",
        "provider": "session-title-first-prompt",
        "model": {"provider": "mock", "model": "mock"},
    }
    request_events = [
        event for event in session.events if event.type == "session/title-llm-request"
    ]
    assert len(request_events) == 1
    assert request_events[0].payload["route"] == {
        "provider": "mock",
        "model": "mock",
    }


async def test_automatic_title_fires_through_stream_turn() -> None:
    """A stream_turn commits the model title from the wired provider row."""
    ctx = create_harness()
    ctx.use(_MockProviderComponent)
    await ctx.flush()
    session = ctx.sessions.create("k5-stream-turn")
    streamed = [
        chunk
        async for chunk in ctx.agent_loop.stream_turn(
            session, "Explain quantum computing simply"
        )
    ]
    assert streamed[-1] == {"type": "turn_end", "text": "Default mock response"}
    await ctx.sessionTitle.drain()
    titles = _provider_title_events(session)
    assert len(titles) == 1
    assert titles[0].payload["title"] == "Default mock response"
    assert titles[0].payload["source"]["provider"] == "session-title-first-prompt"


async def test_disabling_provider_row_restores_fallback() -> None:
    """With the provider row disabled, turns fall back deterministically."""
    ctx = create_harness()
    ctx.use(_MockProviderComponent)
    await ctx.flush()
    loader = ctx.get("loader", None, strict=False)
    await loader.disable("session-title-first-prompt")
    session = ctx.sessions.create("k5-no-provider")
    await ctx.agent_loop.run_turn(session, "Explain quantum computing simply")
    await ctx.sessionTitle.drain()
    titles = [event for event in session.events if event.type == "session/title"]
    assert len(titles) == 1
    assert titles[0].payload["title"] == "Explain quantum computing simply"
    assert titles[0].payload["source"] == {"kind": "fallback"}
    assert _provider_title_events(session) == []


async def test_swapped_new_rows_still_wire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new rows do not depend on bundle order for their dependencies.

    With both new rows moved to the top of the bundle, the components
    slot still resolves and the provider row still waits for the title
    and model services, then titles from the model through a turn.
    """
    rows = _default_rows()
    moved = [
        row for row in rows if row["id"] in ("components", "session-title-first-prompt")
    ]
    rest = [
        row
        for row in rows
        if row["id"] not in ("components", "session-title-first-prompt")
    ]
    monkeypatch.setattr(
        magitech_harness,
        "_BASE_BUNDLE_PATH",
        _write_rows(tmp_path, moved + rest),
    )
    ctx = create_harness()
    ctx.use(_MockProviderComponent)
    await ctx.flush()
    check_bundle_entries(ctx)
    assert ctx.has("components")
    session = ctx.sessions.create("k5-swapped")
    await ctx.agent_loop.run_turn(session, "Explain quantum computing simply")
    await ctx.sessionTitle.drain()
    titles = _provider_title_events(session)
    assert len(titles) == 1
    assert titles[0].payload["title"] == "Default mock response"


async def test_row_mounted_components_settle_on_root_flush(
    tmp_path: Path,
) -> None:
    """A component mounted through the row service is active after one root flush.

    The components service lives on a loader entry context as a row;
    mounting on that entry context would settle only when the entry
    context flushes, which a root flush never awaits, so mounts go to
    the root context instead. This pins that contract: one root flush
    activates the fiber and registers its tool.
    """
    from anima import FiberState

    comp_dir = tmp_path / "k5_probe"
    comp_dir.mkdir()
    (comp_dir / "component.py").write_text(
        "from magitech_harness.tools.definition import define_tool\n"
        "def apply(ctx):\n"
        "    ctx.tools.register(define_tool(\n"
        "        name='k5_probe_add', description='probe',\n"
        "        execute=lambda args: 'probe-ok',\n"
        "    ))\n"
        "apply.inject = ['tools']\n",
        encoding="utf-8",
    )
    (comp_dir / "manifest.json").write_text(
        json.dumps(
            {
                "name": "k5-probe",
                "version": "1.0.0",
                "description": "probe",
                "entry_point": "component.py:apply",
                "types": ["Tool"],
                "enabled": True,
            }
        ),
        encoding="utf-8",
    )
    ctx = create_harness()
    await ctx.flush()
    manifests = ctx.components.discover(tmp_path)
    assert len(manifests) == 1
    fiber = ctx.components.mount_component(manifests[0])
    await ctx.flush()
    assert fiber.state == FiberState.ACTIVE
    assert ctx.tools.get("k5_probe_add") is not None
