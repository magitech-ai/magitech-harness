"""Chunk-level tests for the session-internals-as-rows rework (K3).

The five projection units, the title service, and the checkpoint policy
all mount as bundle rows now; SessionService keeps only the store and
the persistence wiring. These tests pin the chunk's shape at the
bundle level: the units fold from rows, a bad title-row config fails
loud at boot, nothing inside SessionService constructs the moved
pieces, and a directly constructed service still degrades cleanly.
"""

from pathlib import Path

import pytest
from anima import Context

import magitech_harness
from magitech_harness import create_harness
from magitech_harness.session import service as service_module
from magitech_harness.session.checkpoint_policy import apply as apply_checkpoint_policy
from magitech_harness.session.service import SessionService
from magitech_harness.session.title import SessionTitleService


async def test_rows_fold_the_five_units() -> None:
    """The bundle rows fold all five units and feed the title service."""
    ctx = create_harness()
    await ctx.flush()
    service = ctx.sessions
    registry = ctx.sessionProjections
    assert registry.summary_keys() == frozenset({"title", "workingDirectory"})
    session = service.create("k3-1")
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
    service.log_event(session, "step/end", {"turn": 1, "step": 1})
    snapshot = registry.snapshot(session)
    assert snapshot["values"]["turnOutline"] == [
        {"turn": 1, "seq": 0, "prompt": "Hello", "response": ""}
    ]
    assert snapshot["values"]["sessionStats"]["turns"] == 1
    assert snapshot["values"]["workingDirectory"] is None
    # The titleInput unit counted the message the title service reads.
    assert registry.checkpoint(session)["titleInput"]["val"]["count"] == 1
    await ctx.sessionTitle.drain()
    title = ctx.sessionTitle.get(session)
    assert title is not None
    assert title["title"] == "Hello"
    assert title["source"] == {"kind": "fallback"}


def _write_bundle(tmp_path: Path, text: str) -> Path:
    bundle_path = tmp_path / "bundle.yaml"
    bundle_path.write_text(text, encoding="utf-8")
    return bundle_path


_TITLE_BUNDLE = (
    "- id: session-projections\n"
    "  name: magitech_harness.session.projection_registry:SessionProjectionRegistry\n"
    "  config: {}\n"
    "- id: sessions\n"
    "  name: magitech_harness.session.service:SessionService\n"
    "  config: {}\n"
    "- id: session-title\n"
    "  name: magitech_harness.session.title:SessionTitleService\n"
    "  config: {config}\n"
)


@pytest.mark.parametrize(
    "config",
    [
        "{}",
        '{"fallback_max_words": 5, "fallback_max_bytes": 999, "max_title_bytes": 10}',
    ],
)
def test_title_row_config_fails_loud_at_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str
) -> None:
    """A missing or invalid title-row config fails loud, naming the row."""
    bundle_path = _write_bundle(tmp_path, _TITLE_BUNDLE.replace("{config}", config))
    monkeypatch.setattr(magitech_harness, "_BASE_BUNDLE_PATH", bundle_path)
    with pytest.raises(RuntimeError, match="session-title"):
        create_harness()


def test_service_constructs_none_of_the_moved_pieces() -> None:
    """SessionService no longer constructs the registry, title, or policy."""
    source = Path(service_module.__file__).read_text(encoding="utf-8")
    for construction in (
        "SessionTitleService(",
        "SessionCheckpointPolicy(",
        "SessionProjectionRegistry(",
    ):
        assert construction not in source


async def test_direct_construction_degrades_without_projections(
    tmp_path: Path,
) -> None:
    """A bare-ctx service has no cache and fails loud on projection reads."""
    service = SessionService(Context(), {"base_storage_dir": tmp_path})
    assert service.projection_cache is None
    with pytest.raises(RuntimeError, match="sessionProjections"):
        service.read_projections("s1")
    session = service.create("s1")
    service.log_event(session, "turn/start", {"turn": 1})
    await service.flush(session)
    assert service.close("s1") is True
    # A live session without a registry fails loud on the live path too.
    live = SessionService(Context())
    live.create("s1")
    with pytest.raises(RuntimeError, match="sessionProjections"):
        live.read_projections("s1")


def test_service_fails_loud_when_projections_slot_unresolvable() -> None:
    """A context carrying the slot without a value is a loud failure."""
    ctx = Context()
    ctx.provide("sessionProjections", None)
    with pytest.raises(RuntimeError, match="present but unresolvable"):
        SessionService(ctx)


def test_checkpoint_policy_apply_requires_sessions() -> None:
    """The policy wiring fails loud on a context with no sessions service."""
    with pytest.raises(
        RuntimeError, match="session-checkpoint-policy requires the sessions service"
    ):
        apply_checkpoint_policy(Context())


def test_title_service_requires_sessions_and_registry() -> None:
    """The title row fails loud when either injected service is missing."""
    config = {
        "fallback_max_words": 5,
        "fallback_max_bytes": 40,
        "max_title_bytes": 80,
    }
    with pytest.raises(
        RuntimeError, match="session-title requires the sessions service"
    ):
        SessionTitleService(Context(), config)
    ctx = Context()
    SessionService(ctx)
    with pytest.raises(
        RuntimeError, match="session-title requires the sessionProjections service"
    ):
        SessionTitleService(ctx, config)
