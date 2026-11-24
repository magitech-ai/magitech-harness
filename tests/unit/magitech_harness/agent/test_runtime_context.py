"""Unit tests for RuntimeContextProjection and SystemPromptProjection re-exports."""

from __future__ import annotations

from animas import Context

from magitech_harness.agent.runtime_context import (
    RuntimeContextProjection,
    SystemPromptProjection,
)
from magitech_harness.llm.message import create_user_message
from magitech_harness.session.projection_registry import SessionProjectionRegistry
from magitech_harness.session.service import SessionService


def test_runtime_context_projection_lifecycle() -> None:
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    sessions = ctx.sessions
    session = sessions.create("s-rt")
    proj = RuntimeContextProjection(ctx, session)

    assert proj.project("", []) is None

    first = proj.project("pwd: /tmp", [{"name": "cwd", "text": "pwd: /tmp"}])
    assert first is not None
    assert first["source"] == {
        "kind": "runtime-context",
        "form": "snapshot",
        "sections": [{"name": "cwd", "text": "pwd: /tmp"}],
    }
    session.append("user/message", dict(first), surface_op="append")

    # Unchanged snapshot emits nothing
    assert proj.project("pwd: /tmp", [{"name": "cwd", "text": "pwd: /tmp"}]) is None

    # Clearing after non-empty emits the canonical cleared sentinel
    cleared = proj.project("", [])
    assert cleared is not None
    assert cleared["content"] == [
        {
            "type": "text",
            "text": (
                "Current runtime context: none. Earlier runtime-context "
                "snapshots no longer apply."
            ),
        }
    ]
    assert cleared["source"] == {"kind": "runtime-context"}
    ev_cleared = session.append("user/message", dict(cleared), surface_op="append")

    # Identical cleared state is a no-op
    assert proj.project("", []) is None

    # Replacement surface event shadowing the retained node resets retained state
    summary = create_user_message(
        content=[{"type": "text", "text": "compacted"}],
        source={"kind": "user"},
    )
    session.append(
        "user/message",
        dict(summary),
        surface_op={
            "op": "replace",
            "startSeq": ev_cleared.seq,
            "endSeq": ev_cleared.seq,
        },
        source_event_seqs=[ev_cleared.seq],
    )
    after_replace = proj.project("pwd: /repo", [{"name": "cwd", "text": "pwd: /repo"}])
    assert after_replace is not None
    assert after_replace["content"] == [{"type": "text", "text": "pwd: /repo"}]


def test_runtime_context_projection_restores_from_existing_surface_and_ignores_other_sessions() -> (
    None
):
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    sessions = ctx.sessions
    other = sessions.create("s-other")
    s1 = sessions.create("s-restore")

    # Existing snapshot replaced before proj constructs leaves retained = None
    first_msg = create_user_message(
        content=[{"type": "text", "text": "old ctx"}],
        source={"kind": "runtime-context"},
    )
    first_ev = s1.append("user/message", dict(first_msg), surface_op="append")
    replacement = create_user_message(
        content=[{"type": "text", "text": "replaced"}],
        source={"kind": "user"},
    )
    s1.append(
        "user/message",
        dict(replacement),
        surface_op={"op": "replace", "startSeq": first_ev.seq, "endSeq": first_ev.seq},
        source_event_seqs=[first_ev.seq],
    )
    proj_replaced = RuntimeContextProjection(ctx, s1)
    assert proj_replaced.project("", []) is not None

    # Live surviving snapshot on construction is picked up immediately
    s2 = sessions.create("s-surviving")
    live_msg = create_user_message(
        content=[{"type": "text", "text": "alive"}],
        source={"kind": "runtime-context"},
    )
    s2.append("user/message", dict(live_msg), surface_op="append")
    proj_alive = RuntimeContextProjection(ctx, s2)
    assert proj_alive.project("alive", []) is None

    # Events on another session do not alter proj_alive
    other_msg = create_user_message(
        content=[{"type": "text", "text": "other ctx"}],
        source={"kind": "runtime-context"},
    )
    other.append("user/message", dict(other_msg), surface_op="append")
    assert proj_alive.project("alive", []) is None
    # Non-replacement user event on the same session leaves _retained intact
    s2.append(
        "user/message",
        dict(
            create_user_message(
                content=[{"type": "text", "text": "hi"}], source={"kind": "user"}
            )
        ),
        surface_op="append",
    )
    assert proj_alive.project("alive", []) is None
    assert SystemPromptProjection is not None


def test_runtime_context_projection_handles_non_text_and_multiple_replaced_snapshots() -> (
    None
):
    ctx = Context()
    ctx.use(SessionProjectionRegistry)
    ctx.use(SessionService)
    sessions = ctx.sessions

    # Multi-block snapshot yields text=None so empty current triggers cleared sentinel
    s_multi = sessions.create("s-multi")
    multi_msg = create_user_message(
        content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        source={"kind": "runtime-context"},
    )
    s_multi.append("user/message", dict(multi_msg), surface_op="append")
    proj_multi = RuntimeContextProjection(ctx, s_multi)
    assert proj_multi.project("", []) is not None

    # Non-text single block yields text=None as well
    s_non_text = sessions.create("s-non-text")
    s_non_text.append(
        "user/message",
        {
            "role": "user",
            "content": [{"type": "image", "source": {"kind": "url", "url": "x"}}],
            "source": {"kind": "runtime-context"},
        },
        surface_op="append",
    )
    proj_non_text = RuntimeContextProjection(ctx, s_non_text)
    assert proj_non_text.project("now text", []) is not None

    # Multiple replaced snapshots: latest replaced leaves _retained=None when scanning older one
    s_double = sessions.create("s-double")
    ev1 = s_double.append(
        "user/message",
        dict(
            create_user_message(
                content=[{"type": "text", "text": "v1"}],
                source={"kind": "runtime-context"},
            )
        ),
        surface_op="append",
    )
    ev2 = s_double.append(
        "user/message",
        dict(
            create_user_message(
                content=[{"type": "text", "text": "v2"}],
                source={"kind": "runtime-context"},
            )
        ),
        surface_op="append",
    )
    s_double.append(
        "user/message",
        dict(
            create_user_message(
                content=[{"type": "text", "text": "rep"}], source={"kind": "user"}
            )
        ),
        surface_op={"op": "replace", "startSeq": ev1.seq, "endSeq": ev2.seq},
        source_event_seqs=[ev1.seq, ev2.seq],
    )
    proj_double = RuntimeContextProjection(ctx, s_double)
    assert proj_double.project("v2", []) is not None
